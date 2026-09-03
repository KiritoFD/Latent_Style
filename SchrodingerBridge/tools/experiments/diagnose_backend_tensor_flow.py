from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import build_model_from_config  # noqa: E402
from utils.inference import decode_latent, encode_image, load_vae  # noqa: E402


STYLE_SUBDIRS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def pil_to_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    return arr.permute(2, 0, 1) * 2.0 - 1.0


def load_model(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model_from_config(config["model"], use_checkpointing=False).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def edge_map(z: torch.Tensor, kernel: int = 1) -> torch.Tensor:
    ref = z.float()
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        ref = F.avg_pool2d(ref, kernel_size=kernel, stride=1, padding=kernel // 2)
    gx = F.pad(ref[..., :, 1:] - ref[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(ref[..., 1:, :] - ref[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)


def highpass(z: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    if kernel % 2 == 0:
        kernel += 1
    return z.float() - F.avg_pool2d(z.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def lowpass(z: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(z.float(), kernel_size=kernel, stride=1, padding=kernel // 2)


def scalar(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().item())


def q(x: torch.Tensor, p: float) -> float:
    return float(torch.quantile(x.detach().float().reshape(-1), p).item())


def band_energy(z: torch.Tensor) -> dict[str, float]:
    zf = z.float()
    lo = lowpass(zf, 5)
    hi = zf - lo
    return {
        "low_abs_mean": scalar(lo.abs()),
        "high_abs_mean": scalar(hi.abs()),
        "high_to_low_abs": scalar(hi.abs()) / max(scalar(lo.abs()), 1e-12),
        "std": float(zf.std(unbiased=False).item()),
        "abs_p95": q(zf.abs(), 0.95),
        "abs_max": float(zf.abs().max().item()),
    }


def masked_delta_stats(delta: torch.Tensor, edge: torch.Tensor) -> dict[str, float]:
    flat = edge.reshape(edge.shape[0], -1)
    th_hi = torch.quantile(flat, 0.80, dim=1).view(-1, 1, 1, 1)
    th_lo = torch.quantile(flat, 0.20, dim=1).view(-1, 1, 1, 1)
    hi_mask = edge >= th_hi
    lo_mask = edge <= th_lo
    d = delta.float().abs().mean(dim=1, keepdim=True)
    hi = d[hi_mask].mean() if hi_mask.any() else d.new_tensor(float("nan"))
    lo = d[lo_mask].mean() if lo_mask.any() else d.new_tensor(float("nan"))
    all_mean = d.mean()
    return {
        "delta_abs_all": float(all_mean.item()),
        "delta_abs_edge_hi": float(hi.item()),
        "delta_abs_edge_lo": float(lo.item()),
        "edge_hi_to_lo_delta": float((hi / lo.clamp_min(1e-12)).item()) if torch.isfinite(hi) and torch.isfinite(lo) else float("nan"),
        "edge_hi_to_all_delta": float((hi / all_mean.clamp_min(1e-12)).item()) if torch.isfinite(hi) else float("nan"),
    }


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.float().flatten(1)
    bv = b.float().flatten(1)
    return float(F.cosine_similarity(av, bv, dim=1).mean().item())


def image_grid(images: list[torch.Tensor], labels: list[str], path: Path) -> None:
    if not images:
        return
    imgs = [img.detach().cpu().float().clamp(0, 1) for img in images]
    _, h, w = imgs[0].shape
    label_h = 24
    canvas = Image.new("RGB", (w * len(imgs), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(arr), (idx * w, label_h))
        draw.text((idx * w + 4, 5), label[:24], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def diagnose_one(
    *,
    name: str,
    checkpoint: Path,
    eval_dir: Path,
    image_root: Path,
    vae,
    vae_scale: float,
    out_dir: Path,
    device: str,
    image_size: int,
    max_rows: int,
    batch_size: int,
) -> dict:
    model, config = load_model(checkpoint, device)
    style_names = list(config.get("data", {}).get("style_subdirs", STYLE_SUBDIRS))
    style_to_id = {s: i for i, s in enumerate(style_names)}
    model_scale = float(getattr(model, "latent_scale_factor", vae_scale))
    scale_in = model_scale / max(vae_scale, 1e-12)
    scale_out = vae_scale / max(model_scale, 1e-12)

    rows = list(csv.DictReader((eval_dir / "metrics.csv").open("r", encoding="utf-8")))[:max_rows]
    out_rows: list[dict] = []
    first_grid = False

    for start in range(0, len(rows), batch_size):
        br = rows[start : start + batch_size]
        imgs = []
        sids = []
        for row in br:
            imgs.append(pil_to_tensor(image_root / row["src_style"] / row["src_image"], image_size))
            sids.append(style_to_id[row["tgt_style"]])
        img = torch.stack(imgs, dim=0).to(device)
        src01 = (img.float() + 1.0) / 2.0
        z0 = encode_image(vae, img, device=device).float()
        if abs(scale_in - 1.0) > 1e-5:
            z0 = z0 * scale_in
        sid = torch.tensor(sids, dtype=torch.long, device=device)
        delta = model.forward(z0, t=torch.ones((z0.shape[0],), device=device, dtype=z0.dtype), style_id=sid).float()
        zend = z0 + delta
        raw = getattr(model, "last_raw_diffeomorphic", None)
        edge = edge_map(z0, kernel=5)
        hp0 = highpass(z0)
        hp_delta = highpass(delta)
        lp_delta = lowpass(delta)
        if abs(scale_out - 1.0) > 1e-5:
            dec_z = zend * scale_out
        else:
            dec_z = zend
        out01 = decode_latent(vae, dec_z, device=device).float()
        pix_l1 = (out01 - src01).abs().mean(dim=(1, 2, 3))

        if not first_grid:
            image_grid([src01[0], out01[0]], ["source", name], out_dir / name / "tensor_diag_first.png")
            first_grid = True

        for i, row in enumerate(br):
            di = delta[i : i + 1]
            z0i = z0[i : i + 1]
            zendi = zend[i : i + 1]
            hpd = hp_delta[i : i + 1]
            lpd = lp_delta[i : i + 1]
            item = {
                "name": name,
                "src_style": row["src_style"],
                "tgt_style": row["tgt_style"],
                "src_image": row["src_image"],
                "metric_clip_style": row.get("clip_style", ""),
                "metric_content_lpips": row.get("content_lpips", row.get("lpips_content", "")),
                "pixel_l1": float(pix_l1[i].item()),
                "delta_std": float(di.std(unbiased=False).item()),
                "delta_abs_mean": scalar(di.abs()),
                "delta_abs_p95": q(di.abs(), 0.95),
                "delta_abs_max": float(di.abs().max().item()),
                "z0_std": float(z0i.std(unbiased=False).item()),
                "zend_std": float(zendi.std(unbiased=False).item()),
                "zend_to_z0_std_ratio": float(zendi.std(unbiased=False).item() / max(z0i.std(unbiased=False).item(), 1e-12)),
                "high_delta_abs_mean": scalar(hpd.abs()),
                "low_delta_abs_mean": scalar(lpd.abs()),
                "high_to_low_delta_abs": scalar(hpd.abs()) / max(scalar(lpd.abs()), 1e-12),
                "delta_highpass_cos_content_highpass": cosine_flat(hpd, hp0[i : i + 1]),
                "latent_scale_factor": model_scale,
                "scale_in": scale_in,
                "scale_out": scale_out,
                "use_diffeomorphic_stroke": bool(getattr(model, "use_diffeomorphic_stroke", False)),
                "dynamic_style_operator_head": bool(getattr(model, "dynamic_style_operator_head", False)),
                "dynamic_hidden_mult": float(getattr(model, "dynamic_style_operator_hidden_mult", 0.0)),
                "style_spatial_pre_gain_16": float(getattr(model, "style_spatial_pre_gain_16", 0.0)),
                "style_skip_content_retention_boost": float(getattr(model, "style_skip_content_retention_boost", 0.0)),
                "structure_barrier_gamma": float(getattr(model, "structure_barrier_gamma", 0.0)),
            }
            item.update({f"z0_{k}": v for k, v in band_energy(z0i).items()})
            item.update({f"zend_{k}": v for k, v in band_energy(zendi).items()})
            item.update(masked_delta_stats(di, edge[i : i + 1]))
            if raw is not None:
                ri = raw[i : i + 1].float()
                item.update(
                    {
                        "raw_channels": int(ri.shape[1]),
                        "raw_std": float(ri.std(unbiased=False).item()),
                        "raw_abs_mean": scalar(ri.abs()),
                        "raw_abs_p95": q(ri.abs(), 0.95),
                        "raw_abs_max": float(ri.abs().max().item()),
                    }
                )
            out_rows.append(item)

    csv_path = out_dir / name / "tensor_flow_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    numeric_keys = [
        k
        for k, v in out_rows[0].items()
        if isinstance(v, (float, int)) and not isinstance(v, bool)
    ]
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "num_rows": len(out_rows),
        "model_switches": {
            "use_diffeomorphic_stroke": bool(getattr(model, "use_diffeomorphic_stroke", False)),
            "dynamic_style_operator_head": bool(getattr(model, "dynamic_style_operator_head", False)),
            "dynamic_style_operator_hidden_mult": float(getattr(model, "dynamic_style_operator_hidden_mult", 0.0)),
            "style_spatial_pre_gain_16": float(getattr(model, "style_spatial_pre_gain_16", 0.0)),
            "style_skip_content_retention_boost": float(getattr(model, "style_skip_content_retention_boost", 0.0)),
            "structure_barrier_gamma": float(getattr(model, "structure_barrier_gamma", 0.0)),
        },
        "mean": {},
    }
    for key in numeric_keys:
        vals = [float(r[key]) for r in out_rows if r.get(key) == r.get(key)]
        if vals:
            summary["mean"][key] = sum(vals) / len(vals)
    write_json(out_dir / name / "tensor_flow_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, help="name|checkpoint|eval_dir")
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "style_data" / "train")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp" / "diagnostics" / "backend_tensor_flow")
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    vae = load_vae(device=args.device, model_id=args.vae_model, cache_dir=args.cache_dir)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    summaries = []
    for spec in args.spec:
        parts = spec.split("|")
        if len(parts) != 3:
            raise ValueError(f"--spec must be name|checkpoint|eval_dir, got {spec}")
        name, ckpt, eval_dir = parts
        summaries.append(
            diagnose_one(
                name=name,
                checkpoint=Path(ckpt),
                eval_dir=Path(eval_dir),
                image_root=args.image_root,
                vae=vae,
                vae_scale=vae_scale,
                out_dir=args.out_dir,
                device=args.device,
                image_size=args.image_size,
                max_rows=args.max_rows,
                batch_size=args.batch_size,
            )
        )
    write_json(args.out_dir / "summary.json", {"vae_model": args.vae_model, "summaries": summaries})
    print(json.dumps({"out_dir": str(args.out_dir), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
