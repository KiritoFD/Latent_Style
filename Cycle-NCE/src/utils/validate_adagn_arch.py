#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim, affine=False, eps=1e-6)
        self.proj = nn.Linear(style_dim, dim * 2)
        nn.init.normal_(self.proj.weight, std=0.1)
        nn.init.constant_(self.proj.bias, 0.0)
        with torch.no_grad():
            self.proj.bias[:dim] = 1.0

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return h * scale + shift


class SAAdaGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim, affine=False, eps=1e-6)
        self.global_proj = nn.Linear(style_dim, dim * 2)
        nn.init.normal_(self.global_proj.weight, std=0.1)
        nn.init.constant_(self.global_proj.bias, 0.0)
        with torch.no_grad():
            self.global_proj.bias[:dim] = 1.0

        # Lightweight spatial mask generator.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // 4), kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, dim // 4), 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.global_proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        adagn = h * scale + shift
        mask = self.spatial_gate(x)
        return h + mask * (adagn - h)


def calc_high_freq_loss(y: torch.Tensor) -> torch.Tensor:
    dx = y[:, :, :, 1:] - y[:, :, :, :-1]
    dy = y[:, :, 1:, :] - y[:, :, :-1, :]
    return torch.mean(dx.pow(2)) + torch.mean(dy.pow(2))


def run_gradient_probe(device: str) -> None:
    torch.manual_seed(42)
    dim, style_dim, size = 64, 128, 32

    x = torch.randn(1, dim, size, size, device=device)
    x[:, :, 10:20, 10:20] += 5.0
    style_code = torch.randn(1, style_dim, device=device)

    model_std = AdaGN(dim, style_dim, num_groups=4).to(device)
    model_sa = SAAdaGN(dim, style_dim, num_groups=4).to(device)
    model_sa.global_proj.weight.data.copy_(model_std.proj.weight.data)
    model_sa.global_proj.bias.data.copy_(model_std.proj.bias.data)

    style_std = style_code.clone().detach().requires_grad_(True)
    x_std = x.clone().detach().requires_grad_(True)
    y_std = model_std(x_std, style_std)
    loss_std = calc_high_freq_loss(y_std)
    loss_std.backward()

    style_sa = style_code.clone().detach().requires_grad_(True)
    x_sa = x.clone().detach().requires_grad_(True)
    y_sa = model_sa(x_sa, style_sa)
    loss_sa = calc_high_freq_loss(y_sa)
    loss_sa.backward()

    mask_first_conv_grad = list(model_sa.spatial_gate.parameters())[0].grad.norm().item()

    print("== Gradient Probe ==")
    print(f"device: {device}")
    print(f"std   style_grad_norm: {style_std.grad.norm().item():.6f}")
    print(f"sa    style_grad_norm: {style_sa.grad.norm().item():.6f}")
    print(f"std   x_grad_var      : {x_std.grad.var().item():.12e}")
    print(f"sa    x_grad_var      : {x_sa.grad.var().item():.12e}")
    print(f"sa    mask_grad_norm  : {mask_first_conv_grad:.6f}")


def _load_latent_file(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj)
        latent = torch.as_tensor(obj).float()
    elif path.suffix.lower() == ".npy":
        latent = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported latent format: {path}")
    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


def _resolve_data_root(config_path: Path) -> tuple[Path, list[str], int, int]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    rel_root = str(data_cfg.get("data_root", "../../latent-256"))
    style_subdirs = list(data_cfg.get("style_subdirs", ["photo", "Hayao", "monet", "cezanne", "vangogh"]))
    base_dim = int(model_cfg.get("base_dim", 128))
    style_dim = int(model_cfg.get("style_dim", 256))
    data_root = (config_path.parent / rel_root).resolve()
    return data_root, style_subdirs, base_dim, style_dim


def run_real_latent_probe(device: str, config_path: Path, samples_per_style: int) -> None:
    data_root, style_subdirs, dim, style_dim = _resolve_data_root(config_path)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    latent_paths: list[Path] = []
    for sub in style_subdirs:
        sdir = data_root / sub
        if not sdir.exists():
            raise FileNotFoundError(f"style dir not found: {sdir}")
        files = sorted(sdir.glob("*.pt")) + sorted(sdir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no latent files in {sdir}")
        latent_paths.extend(files[: max(1, int(samples_per_style))])

    torch.manual_seed(42)
    stem = nn.Conv2d(4, dim, kernel_size=3, padding=1).to(device)

    model_std = AdaGN(dim, style_dim, num_groups=4).to(device)
    model_sa = SAAdaGN(dim, style_dim, num_groups=4).to(device)
    model_sa.global_proj.weight.data.copy_(model_std.proj.weight.data)
    model_sa.global_proj.bias.data.copy_(model_std.proj.bias.data)

    out_rows = []
    for p in latent_paths:
        z4 = _load_latent_file(p).unsqueeze(0).to(device=device, dtype=torch.float32)
        if z4.shape[1] != 4:
            raise ValueError(f"Expected latent C=4, got {z4.shape[1]} from {p}")
        with torch.no_grad():
            x0 = stem(z4)
        style_code = torch.randn(1, style_dim, device=device)

        s_std = style_code.clone().detach().requires_grad_(True)
        x_std = x0.clone().detach().requires_grad_(True)
        y_std = model_std(x_std, s_std)
        loss_std = calc_high_freq_loss(y_std)
        loss_std.backward()

        s_sa = style_code.clone().detach().requires_grad_(True)
        x_sa = x0.clone().detach().requires_grad_(True)
        y_sa = model_sa(x_sa, s_sa)
        loss_sa = calc_high_freq_loss(y_sa)
        loss_sa.backward()

        row = {
            "style_grad_std": float(s_std.grad.norm().item()),
            "style_grad_sa": float(s_sa.grad.norm().item()),
            "x_grad_var_std": float(x_std.grad.var().item()),
            "x_grad_var_sa": float(x_sa.grad.var().item()),
            "mask_grad_sa": float(list(model_sa.spatial_gate.parameters())[0].grad.norm().item()),
        }
        out_rows.append(row)
        model_sa.zero_grad(set_to_none=True)
        model_std.zero_grad(set_to_none=True)

    def mean(k: str) -> float:
        return sum(r[k] for r in out_rows) / max(1, len(out_rows))

    print("\n== Real Latent Probe (from config data_root) ==")
    print(f"device: {device}")
    print(f"config: {config_path}")
    print(f"data_root: {data_root}")
    print(f"styles: {style_subdirs}")
    print(f"samples: {len(out_rows)}")
    print(f"mean std.style_grad_norm : {mean('style_grad_std'):.6f}")
    print(f"mean sa.style_grad_norm  : {mean('style_grad_sa'):.6f}")
    print(f"mean std.x_grad_var      : {mean('x_grad_var_std'):.12e}")
    print(f"mean sa.x_grad_var       : {mean('x_grad_var_sa'):.12e}")
    print(f"mean sa.mask_grad_norm   : {mean('mask_grad_sa'):.6f}")


def run_conflict_toy(device: str) -> None:
    torch.manual_seed(42)
    dim, size = 64, 32
    h = torch.randn(1, dim, size, size, device=device)
    target = torch.zeros_like(h)
    target[:, :, size // 2 :, :] = torch.randn(1, dim, size // 2, size, device=device) * 5.0

    s_std = torch.nn.Parameter(torch.ones(1, dim, 1, 1, device=device))
    opt_std = torch.optim.Adam([s_std], lr=0.1)
    for _ in range(50):
        y = s_std * h
        loss = F.mse_loss(y[:, :, : size // 2, :], target[:, :, : size // 2, :]) + F.mse_loss(
            y[:, :, size // 2 :, :], target[:, :, size // 2 :, :]
        )
        opt_std.zero_grad()
        loss.backward()
        opt_std.step()

    s_sa = torch.nn.Parameter(torch.ones(1, dim, 1, 1, device=device))
    m_sa = torch.nn.Parameter(torch.ones(1, 1, size, size, device=device) * 0.5)
    opt_sa = torch.optim.Adam([s_sa, m_sa], lr=0.1)
    for _ in range(50):
        y = h + m_sa * (s_sa * h - h)
        loss = F.mse_loss(y[:, :, : size // 2, :], target[:, :, : size // 2, :]) + F.mse_loss(
            y[:, :, size // 2 :, :], target[:, :, size // 2 :, :]
        )
        opt_sa.zero_grad()
        loss.backward()
        opt_sa.step()

    print("\n== Conflict Toy ==")
    print(f"std   scale_mean      : {s_std.mean().item():.6f}")
    print(f"sa    global_scale    : {s_sa.mean().item():.6f}")
    print(f"sa    mask_top_mean   : {m_sa[:, :, : size // 2, :].mean().item():.6f}")
    print(f"sa    mask_bottom_mean: {m_sa[:, :, size // 2 :, :].mean().item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AdaGN vs Spatially-Adaptive AdaGN behavior.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--samples_per_style", type=int, default=1)
    parser.add_argument("--skip_synthetic", action="store_true")
    args = parser.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if not bool(args.skip_synthetic):
        run_gradient_probe(device=device)
        run_conflict_toy(device=device)
    run_real_latent_probe(
        device=device,
        config_path=Path(args.config).resolve(),
        samples_per_style=max(1, int(args.samples_per_style)),
    )


if __name__ == "__main__":
    main()
