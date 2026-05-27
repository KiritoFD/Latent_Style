from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import build_model_from_config  # noqa: E402
from utils.diffeomorphic import _texture_tangent_warp  # noqa: E402
from utils.inference import decode_latent, encode_image, load_vae  # noqa: E402

try:
    import lpips
except Exception:
    lpips = None


STYLE_SUBDIRS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_path(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _image_paths(root: Path, style_names: list[str], max_per_style: int | None = None) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    rng = random.Random(42)
    for style in style_names:
        style_dir = root / style
        if not style_dir.exists():
            continue
        paths = sorted(p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if max_per_style is not None and len(paths) > max_per_style:
            paths = rng.sample(paths, max_per_style)
            paths.sort()
        items.extend((style, p) for p in paths)
    return items


def _latent_paths(root: Path, style_names: list[str], max_per_style: int | None = None) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    rng = random.Random(42)
    for style in style_names:
        style_dir = root / style
        if not style_dir.exists():
            continue
        paths = sorted(style_dir.glob("*.pt"))
        if max_per_style is not None and len(paths) > max_per_style:
            paths = rng.sample(paths, max_per_style)
            paths.sort()
        items.extend((style, p) for p in paths)
    return items


def _pil_to_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    return arr.permute(2, 0, 1) * 2.0 - 1.0


def _load_latent(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("latent", "z", "image", "tensor"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"Unsupported latent file: {path}")
    if obj.ndim == 3:
        obj = obj.unsqueeze(0)
    return obj.float()


def _online_update(stats: dict, x: torch.Tensor) -> None:
    x = x.detach().float().reshape(-1).cpu()
    n = int(x.numel())
    if n <= 0:
        return
    stats["n"] += n
    stats["sum"] += float(x.sum().item())
    stats["sumsq"] += float(x.square().sum().item())
    stats["min"] = min(stats["min"], float(x.min().item()))
    stats["max"] = max(stats["max"], float(x.max().item()))


def _finish_stats(stats: dict) -> dict:
    n = max(1, int(stats["n"]))
    mean = stats["sum"] / n
    var = max(0.0, stats["sumsq"] / n - mean * mean)
    return {
        "num_values": int(stats["n"]),
        "mean": mean,
        "std": math.sqrt(var),
        "min": stats["min"],
        "max": stats["max"],
    }


def _new_stats() -> dict:
    return {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}


@torch.no_grad()
def calibrate_precomputed_latents(latent_root: Path, style_names: list[str], max_per_style: int | None) -> dict:
    per_style = {s: _new_stats() for s in style_names}
    overall = _new_stats()
    counts = defaultdict(int)
    for style, path in _latent_paths(latent_root, style_names, max_per_style=max_per_style):
        z = _load_latent(path)
        _online_update(per_style[style], z)
        _online_update(overall, z)
        counts[style] += 1
    result = {
        "root": str(latent_root),
        "max_per_style": max_per_style,
        "overall": _finish_stats(overall),
        "per_style": {s: _finish_stats(per_style[s]) | {"num_files": counts[s]} for s in style_names},
    }
    std = result["overall"]["std"]
    result["overall"]["scale_to_unit_std"] = 1.0 / max(std, 1e-12)
    return result


@torch.no_grad()
def calibrate_vae_raw_images(
    image_root: Path,
    style_names: list[str],
    *,
    max_per_style: int | None,
    batch_size: int,
    image_size: int,
    device: str,
    vae_model: str,
    cache_dir: str | None,
) -> dict:
    vae = load_vae(device=device, model_id=vae_model, cache_dir=cache_dir)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    per_style_raw = {s: _new_stats() for s in style_names}
    per_style_scaled = {s: _new_stats() for s in style_names}
    overall_raw = _new_stats()
    overall_scaled = _new_stats()
    counts = defaultdict(int)
    batch_imgs: list[torch.Tensor] = []
    batch_styles: list[str] = []

    def flush() -> None:
        if not batch_imgs:
            return
        imgs = torch.stack(batch_imgs, dim=0).to(device)
        raw = vae.encode(imgs.to(dtype=torch.float16)).latent_dist.sample().float()
        scaled = raw * vae_scale
        for i, style in enumerate(batch_styles):
            _online_update(per_style_raw[style], raw[i : i + 1])
            _online_update(per_style_scaled[style], scaled[i : i + 1])
            _online_update(overall_raw, raw[i : i + 1])
            _online_update(overall_scaled, scaled[i : i + 1])
            counts[style] += 1
        batch_imgs.clear()
        batch_styles.clear()

    for style, path in _image_paths(image_root, style_names, max_per_style=max_per_style):
        batch_imgs.append(_pil_to_tensor(path, image_size))
        batch_styles.append(style)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    raw = _finish_stats(overall_raw)
    scaled = _finish_stats(overall_scaled)
    raw["optimal_vae_scale"] = 1.0 / max(raw["std"], 1e-12)
    scaled["scale_to_unit_std"] = 1.0 / max(scaled["std"], 1e-12)
    return {
        "root": str(image_root),
        "image_size": image_size,
        "max_per_style": max_per_style,
        "vae_config_scaling_factor": vae_scale,
        "raw_latent_overall": raw,
        "scaled_latent_overall": scaled,
        "raw_latent_per_style": {
            s: _finish_stats(per_style_raw[s]) | {"num_files": counts[s]} for s in style_names
        },
        "scaled_latent_per_style": {
            s: _finish_stats(per_style_scaled[s]) | {"num_files": counts[s]} for s in style_names
        },
    }


def _load_checkpoint_model(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model_from_config(config["model"], use_checkpointing=False).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def _raw_out_from_forward(model, x: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
    holder: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        holder["raw"] = output.detach()

    handle = model.dec_out.register_forward_hook(hook)
    try:
        _ = model.forward(x, t=t, style_id=style_id)
    finally:
        handle.remove()
    if "raw" not in holder:
        raise RuntimeError("Failed to capture dec_out raw stroke field")
    return holder["raw"]


def _stroke_delta_variant(model, x: torch.Tensor, raw: torch.Tensor, variant: str) -> tuple[torch.Tensor, dict]:
    channels = int(x.shape[1])
    color_strength = float(getattr(model, "diffeomorphic_color_strength", 0.85))
    warp_strength = float(getattr(model, "diffeomorphic_warp_strength", 0.08))
    gate_strength = float(getattr(model, "diffeomorphic_texture_gate_strength", 8.0))
    normal_leak = float(getattr(model, "diffeomorphic_normal_leak", 0.0))
    color_lowpass_kernel = max(1, int(getattr(model, "diffeomorphic_color_lowpass_kernel", 1)))
    color_edge_gamma = max(0.0, float(getattr(model, "diffeomorphic_color_edge_gamma", 0.0)))

    raw_color = raw[:, :channels]
    raw_warp = raw[:, channels : channels + 2]
    color_delta = torch.tanh(raw_color) * color_strength
    if color_lowpass_kernel > 1:
        if color_lowpass_kernel % 2 == 0:
            color_lowpass_kernel += 1
        color_delta = F.avg_pool2d(
            color_delta,
            kernel_size=color_lowpass_kernel,
            stride=1,
            padding=color_lowpass_kernel // 2,
        )
    if color_edge_gamma > 0.0:
        edge_dx = F.pad(x.float()[:, :, :, 1:] - x.float()[:, :, :, :-1], (0, 1, 0, 0))
        edge_dy = F.pad(x.float()[:, :, 1:, :] - x.float()[:, :, :-1, :], (0, 0, 0, 1))
        edge_mag = torch.sqrt(edge_dx.square() + edge_dy.square() + 1e-12).mean(dim=1, keepdim=True)
        color_delta = color_delta * torch.exp(-color_edge_gamma * edge_mag)
    spatial_warp = torch.tanh(raw_warp) * warp_strength
    effective_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
    )
    if variant == "disable_color":
        color_delta = torch.zeros_like(color_delta)
    if variant == "disable_warp":
        effective_warp = torch.zeros_like(effective_warp)
    if variant == "disable_both":
        color_delta = torch.zeros_like(color_delta)
        effective_warp = torch.zeros_like(effective_warp)

    b, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    warped_grid = (base_grid + effective_warp.permute(0, 2, 3, 1)).clamp(-1.2, 1.2)
    x_warped = F.grid_sample(x.float(), warped_grid, align_corners=False, padding_mode="reflection")
    delta = x_warped + color_delta - x.float()
    stats = {
        "raw_color_abs_max": float(raw_color.abs().max().item()),
        "raw_color_abs_mean": float(raw_color.abs().mean().item()),
        "raw_warp_abs_max": float(raw_warp.abs().max().item()),
        "raw_warp_abs_mean": float(raw_warp.abs().mean().item()),
        "color_delta_abs_max": float(color_delta.abs().max().item()),
        "color_delta_abs_mean": float(color_delta.abs().mean().item()),
        "effective_warp_abs_max": float(effective_warp.abs().max().item()),
        "effective_warp_abs_mean": float(effective_warp.abs().mean().item()),
        "delta_abs_max": float(delta.abs().max().item()),
        "delta_abs_mean": float(delta.abs().mean().item()),
        "color_lowpass_kernel": float(color_lowpass_kernel),
        "color_edge_gamma": float(color_edge_gamma),
    }
    stats.update({f"effective_{k}": v for k, v in _jacobian_folding_stats(effective_warp).items()})
    stats.update({f"raw_spatial_{k}": v for k, v in _jacobian_folding_stats(spatial_warp).items()})
    return delta, stats


def _lpips_input(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _mean_dict(rows: list[dict], keys: Iterable[str]) -> dict:
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        if vals:
            out[key] = sum(vals) / len(vals)
    out["count"] = len(rows)
    return out


def _jacobian_folding_stats(warp: torch.Tensor) -> dict:
    if warp.ndim != 4 or warp.shape[1] != 2:
        return {
            "folding_ratio": float("nan"),
            "jacobian_det_min": float("nan"),
            "jacobian_det_p01": float("nan"),
            "jacobian_det_mean": float("nan"),
        }
    wx = warp[:, 0:1].float()
    wy = warp[:, 1:2].float()
    dwx_dx = F.pad(wx[:, :, :, 1:] - wx[:, :, :, :-1], (0, 1, 0, 0))
    dwy_dx = F.pad(wy[:, :, :, 1:] - wy[:, :, :, :-1], (0, 1, 0, 0))
    dwx_dy = F.pad(wx[:, :, 1:, :] - wx[:, :, :-1, :], (0, 0, 0, 1))
    dwy_dy = F.pad(wy[:, :, 1:, :] - wy[:, :, :-1, :], (0, 0, 0, 1))
    det_j = (1.0 + dwx_dx) * (1.0 + dwy_dy) - dwx_dy * dwy_dx
    det_flat = det_j.reshape(-1)
    return {
        "folding_ratio": float((det_j <= 0).float().mean().item()),
        "jacobian_det_min": float(det_flat.min().item()),
        "jacobian_det_p01": float(torch.quantile(det_flat, 0.01).item()),
        "jacobian_det_mean": float(det_flat.mean().item()),
    }


def _latent_norm_stats(z: torch.Tensor) -> dict:
    zf = z.detach().float()
    channel_norm = torch.linalg.vector_norm(zf, dim=1).mean()
    sample_norm = torch.linalg.vector_norm(zf.flatten(1), dim=1).mean()
    return {
        "latent_channel_norm_mean": float(channel_norm.item()),
        "latent_sample_norm_mean": float(sample_norm.item()),
        "latent_std": float(zf.std(unbiased=False).item()),
        "latent_abs_max": float(zf.abs().max().item()),
    }


def _make_grid_with_labels(images: list[torch.Tensor], labels: list[str], path: Path) -> None:
    if not images:
        return
    imgs = [img.detach().cpu().float().clamp(0, 1) for img in images]
    c, h, w = imgs[0].shape
    label_h = 22
    canvas = Image.new("RGB", (w * len(imgs), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(arr), (i * w, label_h))
        draw.text((i * w + 4, 4), label[:22], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def run_decoupling(
    checkpoint: Path,
    eval_dir: Path,
    image_root: Path,
    out_dir: Path,
    *,
    max_rows: int | None,
    batch_size: int,
    image_size: int,
    device: str,
    vae_model: str,
    cache_dir: str | None,
) -> dict:
    model, config = _load_checkpoint_model(checkpoint, device)
    style_names = list(config.get("data", {}).get("style_subdirs", STYLE_SUBDIRS))
    style_to_id = {s: i for i, s in enumerate(style_names)}
    vae = load_vae(device=device, model_id=vae_model, cache_dir=cache_dir)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    model_scale = float(getattr(model, "latent_scale_factor", vae_scale))
    scale_in = model_scale / max(vae_scale, 1e-12)
    scale_out = vae_scale / max(model_scale, 1e-12)
    metrics_path = eval_dir / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    if max_rows is not None:
        rows = rows[:max_rows]

    loss_fn = lpips.LPIPS(net="alex").to(device).eval() if lpips is not None else None
    variants = ["normal", "disable_warp", "disable_color", "disable_both"]
    results: list[dict] = []
    sample_grid_done = False

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        imgs = []
        style_ids = []
        for row in batch_rows:
            src = image_root / row["src_style"] / row["src_image"]
            imgs.append(_pil_to_tensor(src, image_size))
            style_ids.append(style_to_id[row["tgt_style"]])
        img_tensor = torch.stack(imgs, dim=0).to(device)
        z0 = encode_image(vae, img_tensor, device=device).float()
        if abs(scale_in - 1.0) > 1e-5:
            z0 = z0 * scale_in
        sid = torch.tensor(style_ids, dtype=torch.long, device=device)
        t = torch.ones((z0.shape[0],), dtype=z0.dtype, device=device)
        raw = _raw_out_from_forward(model, z0, t, sid)
        src01 = (img_tensor.float() + 1.0) / 2.0

        decoded_by_variant: dict[str, torch.Tensor] = {}
        for variant in variants:
            delta, stat = _stroke_delta_variant(model, z0, raw, variant)
            zout = z0 + delta
            if abs(scale_out - 1.0) > 1e-5:
                zout = zout * scale_out
            out = decode_latent(vae, zout, device=device).float()
            decoded_by_variant[variant] = out
            pix_l1 = (out - src01).abs().mean(dim=(1, 2, 3)).detach().cpu()
            latent_l1 = delta.abs().mean(dim=(1, 2, 3)).detach().cpu()
            if loss_fn is not None:
                lp = loss_fn(_lpips_input(out), _lpips_input(src01)).detach().cpu().view(-1)
            else:
                lp = torch.full((out.shape[0],), float("nan"))
            for i, row in enumerate(batch_rows):
                item = {
                    "variant": variant,
                    "src_style": row["src_style"],
                    "tgt_style": row["tgt_style"],
                    "src_image": row["src_image"],
                    "lpips_content": float(lp[i].item()),
                    "pixel_l1": float(pix_l1[i].item()),
                    "latent_delta_l1": float(latent_l1[i].item()),
                }
                item.update(stat)
                results.append(item)

        if not sample_grid_done:
            grid_images = [src01[0].detach().cpu()]
            labels = ["source"]
            for variant in variants:
                grid_images.append(decoded_by_variant[variant][0].detach().cpu())
                labels.append(variant)
            _make_grid_with_labels(grid_images, labels, out_dir / "decoupling_sample_grid.png")
            sample_grid_done = True

    csv_path = out_dir / "decoupling_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "image_root": str(image_root),
        "num_pairs": len(rows),
        "vae_config_scaling_factor": vae_scale,
        "model_latent_scale_factor": model_scale,
        "scale_in": scale_in,
        "scale_out": scale_out,
        "by_variant": {},
        "by_target_style": {},
    }
    metric_keys = [
        "lpips_content",
        "pixel_l1",
        "latent_delta_l1",
        "raw_color_abs_max",
        "raw_warp_abs_max",
        "effective_warp_abs_max",
        "color_delta_abs_mean",
        "effective_warp_abs_mean",
        "delta_abs_mean",
        "effective_folding_ratio",
        "effective_jacobian_det_min",
        "effective_jacobian_det_p01",
        "raw_spatial_folding_ratio",
        "raw_spatial_jacobian_det_min",
    ]
    for variant in variants:
        vr = [r for r in results if r["variant"] == variant]
        summary["by_variant"][variant] = _mean_dict(vr, metric_keys)
    for variant in variants:
        summary["by_target_style"][variant] = {}
        for style in style_names:
            sr = [r for r in results if r["variant"] == variant and r["tgt_style"] == style]
            if sr:
                summary["by_target_style"][variant][style] = _mean_dict(sr, metric_keys)
    _write_json(out_dir / "decoupling_summary.json", summary)
    return summary


@torch.no_grad()
def run_trajectory_audit(
    checkpoint: Path,
    eval_dir: Path,
    image_root: Path,
    out_dir: Path,
    *,
    num_samples: int,
    num_steps: int,
    image_size: int,
    device: str,
    vae_model: str,
    cache_dir: str | None,
) -> dict:
    model, config = _load_checkpoint_model(checkpoint, device)
    style_names = list(config.get("data", {}).get("style_subdirs", STYLE_SUBDIRS))
    style_to_id = {s: i for i, s in enumerate(style_names)}
    vae = load_vae(device=device, model_id=vae_model, cache_dir=cache_dir)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    model_scale = float(getattr(model, "latent_scale_factor", vae_scale))
    scale_in = model_scale / max(vae_scale, 1e-12)
    scale_out = vae_scale / max(model_scale, 1e-12)
    rows = list(csv.DictReader((eval_dir / "metrics.csv").open("r", encoding="utf-8")))[:num_samples]
    logs: list[dict] = []

    for sample_idx, row in enumerate(rows):
        src = image_root / row["src_style"] / row["src_image"]
        img = _pil_to_tensor(src, image_size).unsqueeze(0).to(device)
        z = encode_image(vae, img, device=device).float()
        if abs(scale_in - 1.0) > 1e-5:
            z = z * scale_in
        z0_stats = _latent_norm_stats(z)
        sid = torch.tensor([style_to_id[row["tgt_style"]]], dtype=torch.long, device=device)
        horizon = 1.0
        dt = horizon / float(num_steps)
        decoded_steps: list[torch.Tensor] = []
        labels: list[str] = []
        for step in range(num_steps + 1):
            zout = z * scale_out if abs(scale_out - 1.0) > 1e-5 else z
            decoded_steps.append(decode_latent(vae, zout, device=device)[0].detach().cpu())
            labels.append(f"s{step:02d}")
            if step == num_steps:
                break
            t = torch.full((1,), horizon * ((step + 0.5) / num_steps), dtype=z.dtype, device=device)
            raw = _raw_out_from_forward(model, z, t, sid)
            delta, stat = _stroke_delta_variant(model, z, raw, "normal")
            z = z + delta * dt
            z_stats = _latent_norm_stats(z)
            stat.update(
                {
                    "sample_idx": sample_idx,
                    "step": step,
                    "src_style": row["src_style"],
                    "tgt_style": row["tgt_style"],
                    "src_image": row["src_image"],
                    "latent_channel_norm_initial": z0_stats["latent_channel_norm_mean"],
                    "latent_sample_norm_initial": z0_stats["latent_sample_norm_mean"],
                    "latent_std_initial": z0_stats["latent_std"],
                    "latent_channel_norm_after": z_stats["latent_channel_norm_mean"],
                    "latent_sample_norm_after": z_stats["latent_sample_norm_mean"],
                    "latent_std_after": z_stats["latent_std"],
                    "latent_abs_max_after": z_stats["latent_abs_max"],
                    "latent_channel_norm_drift_ratio": (
                        z_stats["latent_channel_norm_mean"] / max(z0_stats["latent_channel_norm_mean"], 1e-12) - 1.0
                    ),
                    "latent_sample_norm_drift_ratio": (
                        z_stats["latent_sample_norm_mean"] / max(z0_stats["latent_sample_norm_mean"], 1e-12) - 1.0
                    ),
                }
            )
            logs.append(stat)
        _make_grid_with_labels(
            decoded_steps,
            labels,
            out_dir / "trajectory" / f"sample_{sample_idx:02d}_{row['src_style']}_to_{row['tgt_style']}.png",
        )

    csv_path = out_dir / "trajectory_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)
    summary = {
        "checkpoint": str(checkpoint),
        "num_samples": len(rows),
        "num_steps": num_steps,
        "max_raw_warp_abs": max(r["raw_warp_abs_max"] for r in logs),
        "max_effective_warp_abs": max(r["effective_warp_abs_max"] for r in logs),
        "max_raw_color_abs": max(r["raw_color_abs_max"] for r in logs),
        "max_delta_abs": max(r["delta_abs_max"] for r in logs),
        "max_effective_folding_ratio": max(r["effective_folding_ratio"] for r in logs),
        "min_effective_jacobian_det": min(r["effective_jacobian_det_min"] for r in logs),
        "max_raw_spatial_folding_ratio": max(r["raw_spatial_folding_ratio"] for r in logs),
        "min_raw_spatial_jacobian_det": min(r["raw_spatial_jacobian_det_min"] for r in logs),
        "max_latent_channel_norm_drift_ratio": max(r["latent_channel_norm_drift_ratio"] for r in logs),
        "max_latent_sample_norm_drift_ratio": max(r["latent_sample_norm_drift_ratio"] for r in logs),
        "mean_effective_warp_abs": sum(r["effective_warp_abs_mean"] for r in logs) / len(logs),
        "mean_color_delta_abs": sum(r["color_delta_abs_mean"] for r in logs) / len(logs),
    }
    _write_json(out_dir / "trajectory_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only diagnostics for LANCET diffeomorphic tangent models.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp/diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/epoch_0008.pt")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "exp/diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/full_eval/epoch_0008")
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "style_data/overfit50")
    parser.add_argument("--latent-root", type=Path, default=ROOT.parent / "latent-256")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/forward_diagnostics/t00_epoch8")
    parser.add_argument("--style-subdirs", type=str, default=",".join(STYLE_SUBDIRS))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-model", type=str, default="sd15")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vae-per-style", type=int, default=50)
    parser.add_argument("--max-latent-per-style", type=int, default=None)
    parser.add_argument("--max-decoupling-rows", type=int, default=None)
    parser.add_argument("--trajectory-samples", type=int, default=4)
    parser.add_argument("--trajectory-steps", type=int, default=16)
    parser.add_argument("--skip-vae-raw", action="store_true")
    parser.add_argument("--skip-latent-stats", action="store_true")
    parser.add_argument("--skip-decoupling", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_subdirs.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "eval_dir": str(args.eval_dir),
        "image_root": str(args.image_root),
        "latent_root": str(args.latent_root),
        "style_subdirs": style_names,
    }

    if not args.skip_latent_stats:
        latent_stats = calibrate_precomputed_latents(args.latent_root, style_names, args.max_latent_per_style)
        _write_json(args.out_dir / "latent256_scale_stats.json", latent_stats)
        manifest["latent256_scale_stats"] = latent_stats
        print("[latent] overall std:", latent_stats["overall"]["std"], "scale_to_unit:", latent_stats["overall"]["scale_to_unit_std"])

    if not args.skip_vae_raw:
        vae_stats = calibrate_vae_raw_images(
            args.image_root,
            style_names,
            max_per_style=args.max_vae_per_style,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            vae_model=args.vae_model,
            cache_dir=args.cache_dir,
        )
        _write_json(args.out_dir / "vae_raw_scale_calibration.json", vae_stats)
        manifest["vae_raw_scale_calibration"] = vae_stats
        print(
            "[vae-raw] std:",
            vae_stats["raw_latent_overall"]["std"],
            "optimal_scale:",
            vae_stats["raw_latent_overall"]["optimal_vae_scale"],
        )

    if not args.skip_decoupling:
        dec = run_decoupling(
            args.checkpoint,
            args.eval_dir,
            args.image_root,
            args.out_dir,
            max_rows=args.max_decoupling_rows,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            vae_model=args.vae_model,
            cache_dir=args.cache_dir,
        )
        manifest["decoupling_summary"] = dec
        print("[decoupling]", json.dumps(dec["by_variant"], indent=2))

    if not args.skip_trajectory:
        traj = run_trajectory_audit(
            args.checkpoint,
            args.eval_dir,
            args.image_root,
            args.out_dir,
            num_samples=args.trajectory_samples,
            num_steps=args.trajectory_steps,
            image_size=args.image_size,
            device=args.device,
            vae_model=args.vae_model,
            cache_dir=args.cache_dir,
        )
        manifest["trajectory_summary"] = traj
        print("[trajectory]", json.dumps(traj, indent=2))

    _write_json(args.out_dir / "summary.json", manifest)
    print("Saved diagnostics to", args.out_dir)


if __name__ == "__main__":
    main()
