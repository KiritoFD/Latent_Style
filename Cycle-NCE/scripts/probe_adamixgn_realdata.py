from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGN(nn.Module):
    def __init__(self, dim: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, dim, affine=False)
        self.proj = nn.Linear(style_dim, dim * 2)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)
        with torch.no_grad():
            self.proj.bias[:dim] = 1.0

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.proj(style_code).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return h * scale + shift


class AdaMixGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, rank: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, dim, affine=False)
        self.style_shift = nn.Linear(style_dim, dim)
        self.rank = rank
        self.style_U = nn.Linear(style_dim, dim * rank)
        self.style_V = nn.Linear(style_dim, rank * dim)

        nn.init.zeros_(self.style_shift.weight)
        nn.init.zeros_(self.style_shift.bias)
        nn.init.normal_(self.style_U.weight, std=0.01)
        nn.init.zeros_(self.style_U.bias)
        nn.init.zeros_(self.style_V.weight)
        nn.init.zeros_(self.style_V.bias)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        normalized = self.norm(x)

        shift = self.style_shift(style_code).view(b, c, 1, 1)
        u = self.style_U(style_code).view(b, c, self.rank)
        v = self.style_V(style_code).view(b, self.rank, c)
        w = torch.bmm(u, v)

        norm_flat = normalized.view(b, c, h_dim * w_dim)
        mixed_flat = torch.bmm(w, norm_flat)
        mixed = mixed_flat.view(b, c, h_dim, w_dim)
        return normalized + mixed + shift


def _resolve_data_root(config_path: Path) -> Tuple[Path, list[str], int]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    root_raw = Path(data_cfg.get("data_root", "../../latent-256"))
    data_root = (config_path.parent / root_raw).resolve() if not root_raw.is_absolute() else root_raw.resolve()
    styles = list(data_cfg.get("style_subdirs", ["photo", "Hayao", "monet", "vangogh", "cezanne"]))
    style_dim = int(model_cfg.get("style_dim", 256))
    return data_root, styles, style_dim


def _load_batch(data_root: Path, style_name: str, max_samples: int, device: torch.device) -> torch.Tensor:
    style_dir = data_root / style_name
    files = sorted(style_dir.glob("*.pt"))[:max_samples]
    if not files:
        raise RuntimeError(f"No .pt latents found: {style_dir}")
    latents = []
    for p in files:
        x = torch.load(p, map_location="cpu")
        x = torch.as_tensor(x).float()
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected latent shape {tuple(x.shape)} in {p}")
        latents.append(x)
    return torch.stack(latents, dim=0).to(device=device)


def _pick_channels(x: torch.Tensor) -> Tuple[int, int, int]:
    # Dead-water channel: lowest overall energy.
    channel_energy = x.pow(2).mean(dim=(0, 2, 3))
    sink = int(torch.argmin(channel_energy).item())
    # High-frequency channels: strongest spatial variation.
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    hf = grad_x.abs().mean(dim=(0, 2, 3)) + grad_y.abs().mean(dim=(0, 2, 3))
    top = torch.argsort(hf, descending=True).tolist()
    sources = [c for c in top if c != sink]
    if len(sources) < 2:
        raise RuntimeError(f"Need >=2 source channels, got {sources}")
    return sink, int(sources[0]), int(sources[1])


def _train_and_eval(
    module_class: type[nn.Module],
    name: str,
    x: torch.Tensor,
    style_dim: int,
    sink: int,
    src_a: int,
    src_b: int,
    steps: int,
    lr: float,
    rank: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    bsz, dim, _, _ = x.shape

    if module_class is AdaMixGN:
        model = module_class(dim, style_dim, rank=rank).to(x.device)
    else:
        model = module_class(dim, style_dim).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    target_a = x[:, src_a].clone()
    target_b = x[:, src_b].clone()
    style_code_a = torch.randn(bsz, style_dim, device=x.device)
    style_code_b = torch.randn(bsz, style_dim, device=x.device)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        y_a = model(x, style_code_a)
        y_b = model(x, style_code_b)
        loss_a = F.mse_loss(y_a[:, sink], target_a)
        loss_b = F.mse_loss(y_b[:, sink], target_b)
        loss = loss_a + loss_b
        loss.backward()
        opt.step()

    with torch.no_grad():
        y_a = model(x, style_code_a)
        y_b = model(x, style_code_b)
        mse_a = F.mse_loss(y_a[:, sink], target_a).item()
        mse_b = F.mse_loss(y_b[:, sink], target_b).item()
        signal_var = target_a.var(unbiased=False)
        noise_var = (target_a - y_a[:, sink]).var(unbiased=False)
        snr_db = (10.0 * torch.log10(signal_var / (noise_var + 1e-8))).item()

    print(f"\n--- {name} ---")
    print(f"Sink channel={sink}, Source-A={src_a}, Source-B={src_b}")
    print(f"MSE(A): {mse_a:.6f}")
    print(f"MSE(B): {mse_b:.6f}")
    print(f"SNR(A): {snr_db:.2f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data cross-channel routing probe for AdaGN vs AdaMixGN")
    parser.add_argument("--config", type=str, default="src/config.json")
    parser.add_argument("--style", type=str, default="photo", help="Style domain to sample real latents from")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    data_root, styles, style_dim = _resolve_data_root(cfg_path)
    if args.style not in styles:
        raise ValueError(f"--style={args.style} not in config data.style_subdirs={styles}")
    device = torch.device(args.device)

    x = _load_batch(data_root, args.style, args.max_samples, device)
    sink, src_a, src_b = _pick_channels(x)

    print("=== Real-data Micro-probe ===")
    print(f"config: {cfg_path}")
    print(f"data_root: {data_root}")
    print(f"style: {args.style}")
    print(f"batch: {x.shape[0]}, latent_shape: {tuple(x.shape[1:])}, style_dim: {style_dim}")

    _train_and_eval(
        AdaGN,
        "Standard AdaGN (per-channel affine)",
        x,
        style_dim,
        sink,
        src_a,
        src_b,
        args.steps,
        args.lr,
        args.rank,
        args.seed,
    )
    _train_and_eval(
        AdaMixGN,
        f"AdaMixGN (low-rank dynamic mixing, rank={args.rank})",
        x,
        style_dim,
        sink,
        src_a,
        src_b,
        args.steps,
        args.lr,
        args.rank,
        args.seed,
    )


if __name__ == "__main__":
    main()
