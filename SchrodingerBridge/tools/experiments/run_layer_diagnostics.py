from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import build_model_from_config  # noqa: E402
from utils.inference import encode_image, load_vae  # noqa: E402


STYLE_SUBDIRS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _pil_to_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    return arr.permute(2, 0, 1) * 2.0 - 1.0


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


def _new_acc() -> dict[str, float]:
    return {
        "count": 0.0,
        "mean_sum": 0.0,
        "std_sum": 0.0,
        "abs_mean_sum": 0.0,
        "abs_max_max": 0.0,
        "hf_ratio_sum": 0.0,
    }


def _stats_tensor(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    if x.ndim < 2:
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "abs_mean": float(x.abs().mean().item()),
            "abs_max": float(x.abs().max().item()),
            "hf_ratio": 0.0,
        }
    hf_ratio = 0.0
    if x.ndim == 4 and min(x.shape[-2:]) >= 3:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        hf = x - low
        hf_ratio = float((hf.square().mean() / (x.square().mean() + 1e-12)).item())
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "abs_max": float(x.abs().max().item()),
        "hf_ratio": hf_ratio,
    }


def _update(acc: dict[str, float], stats: dict[str, float]) -> None:
    acc["count"] += 1.0
    acc["mean_sum"] += stats["mean"]
    acc["std_sum"] += stats["std"]
    acc["abs_mean_sum"] += stats["abs_mean"]
    acc["abs_max_max"] = max(acc["abs_max_max"], stats["abs_max"])
    acc["hf_ratio_sum"] += stats["hf_ratio"]


def _finish(acc: dict[str, float]) -> dict[str, float]:
    n = max(acc["count"], 1.0)
    return {
        "count": int(acc["count"]),
        "mean": acc["mean_sum"] / n,
        "std": acc["std_sum"] / n,
        "abs_mean": acc["abs_mean_sum"] / n,
        "abs_max": acc["abs_max_max"],
        "hf_ratio": acc["hf_ratio_sum"] / n,
    }


def _module_specs(model) -> list[tuple[str, torch.nn.Module]]:
    specs: list[tuple[str, torch.nn.Module]] = []
    specs.append(("enc_in", model.enc_in))
    specs.append(("enc_in_act", model.enc_in_act))
    for idx, block in enumerate(model.hires_body):
        specs.append((f"hires_body.{idx}", block))
    specs.append(("down", model.down))
    for idx, block in enumerate(model.body_blocks):
        specs.append((f"body_blocks.{idx}", block))
    specs.append(("dec_up", model.dec_up))
    specs.append(("skip_up_proj", model.skip_up_proj))
    specs.append(("skip_src_proj", model.skip_src_proj))
    specs.append(("skip_fusion", model.skip_fusion))
    for idx, block in enumerate(model.decoder_blocks):
        specs.append((f"decoder_blocks.{idx}", block))
    specs.append(("dec_out", model.dec_out))
    return specs


@torch.no_grad()
def run_one(
    *,
    name: str,
    checkpoint: Path,
    eval_dir: Path,
    image_root: Path,
    out_dir: Path,
    max_rows: int | None,
    batch_size: int,
    image_size: int,
    device: str,
    vae_model: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    model, config = _load_checkpoint_model(checkpoint, device)
    style_names = list(config.get("data", {}).get("style_subdirs", STYLE_SUBDIRS))
    style_to_id = {s: i for i, s in enumerate(style_names)}
    vae = load_vae(device=device, model_id=vae_model, cache_dir=cache_dir)
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    model_scale = float(getattr(model, "latent_scale_factor", vae_scale))
    scale_in = model_scale / max(vae_scale, 1e-12)

    rows = list(csv.DictReader((eval_dir / "metrics.csv").open("r", encoding="utf-8")))
    if max_rows is not None:
        rows = rows[:max_rows]

    layer_acc: dict[str, dict[str, float]] = defaultdict(_new_acc)
    layer_style_acc: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(_new_acc))
    head_acc: dict[str, dict[str, float]] = defaultdict(_new_acc)
    head_style_acc: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(_new_acc))
    current_styles: list[str] = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                return
            stats = _stats_tensor(tensor)
            _update(layer_acc[layer_name], stats)
            if tensor.ndim >= 1 and tensor.shape[0] == len(current_styles):
                for idx, style in enumerate(current_styles):
                    _update(layer_style_acc[layer_name][style], _stats_tensor(tensor[idx : idx + 1]))
            if layer_name == "dec_out" and tensor.ndim == 4:
                channels = int(getattr(model, "latent_channels", 4))
                parts = {
                    "raw_color": tensor[:, :channels],
                    "raw_warp": tensor[:, channels : channels + 2],
                }
                for part_name, part in parts.items():
                    _update(head_acc[part_name], _stats_tensor(part))
                    if part.shape[0] == len(current_styles):
                        for idx, style in enumerate(current_styles):
                            _update(head_style_acc[part_name][style], _stats_tensor(part[idx : idx + 1]))
        return hook

    handles = [module.register_forward_hook(make_hook(layer_name)) for layer_name, module in _module_specs(model)]
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            imgs = []
            style_ids = []
            current_styles = []
            for row in batch_rows:
                src = image_root / row["src_style"] / row["src_image"]
                imgs.append(_pil_to_tensor(src, image_size))
                style_ids.append(style_to_id[row["tgt_style"]])
                current_styles.append(row["tgt_style"])
            img_tensor = torch.stack(imgs, dim=0).to(device)
            z0 = encode_image(vae, img_tensor, device=device).float()
            if abs(scale_in - 1.0) > 1e-5:
                z0 = z0 * scale_in
            sid = torch.tensor(style_ids, dtype=torch.long, device=device)
            try:
                _ = model.forward(z0, t=torch.ones((z0.shape[0],), dtype=z0.dtype, device=device), style_id=sid)
            except TypeError:
                _ = model.forward(z0, style_id=sid)
    finally:
        for handle in handles:
            handle.remove()

    payload = {
        "name": name,
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "num_pairs": len(rows),
        "layers": {k: _finish(v) for k, v in sorted(layer_acc.items())},
        "heads": {k: _finish(v) for k, v in sorted(head_acc.items())},
        "layers_by_target_style": {
            layer: {style: _finish(acc) for style, acc in sorted(style_map.items())}
            for layer, style_map in sorted(layer_style_acc.items())
        },
        "heads_by_target_style": {
            head: {style: _finish(acc) for style, acc in sorted(style_map.items())}
            for head, style_map in sorted(head_style_acc.items())
        },
    }
    _write_json(out_dir / f"{name}_layer_trace.json", payload)
    return payload


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


def build_report(results: list[dict[str, Any]], out_dir: Path) -> None:
    by_name = {r["name"]: r for r in results}
    baseline = by_name.get("t00")
    lines = [
        "# 2026-05-21 Layer Diagnostics",
        "",
        "Purpose: locate where the color-guard variants diverge from the stable `t00` tangent baseline.",
        "",
        "## Head Summary",
        "",
        "| model | raw color abs max | raw color abs mean | raw color HF | raw warp abs max | raw warp abs mean | raw warp HF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        color = r["heads"].get("raw_color", {})
        warp = r["heads"].get("raw_warp", {})
        lines.append(
            f"| {r['name']} | {color.get('abs_max', 0):.4f} | {color.get('abs_mean', 0):.4f} | "
            f"{color.get('hf_ratio', 0):.4f} | {warp.get('abs_max', 0):.4f} | "
            f"{warp.get('abs_mean', 0):.4f} | {warp.get('hf_ratio', 0):.4f} |"
        )
    lines.extend(["", "## Layer Ratios vs t00", ""])
    if baseline is not None:
        keys = ["content_feat_16", "body_blocks.0", "body_blocks.3", "skip_fusion", "decoder_blocks.0", "decoder_blocks.1", "dec_out"]
        aliases = {"content_feat_16": "down"}
        lines.append("| layer | model | abs mean ratio | HF ratio ratio | abs max ratio |")
        lines.append("|---|---|---:|---:|---:|")
        for layer in keys:
            actual = aliases.get(layer, layer)
            base_stats = baseline["layers"].get(actual, {})
            for r in results:
                if r["name"] == "t00":
                    continue
                stats = r["layers"].get(actual, {})
                abs_mean_ratio = _ratio(stats.get("abs_mean"), base_stats.get("abs_mean"))
                hf_ratio = _ratio(stats.get("hf_ratio"), base_stats.get("hf_ratio"))
                abs_max_ratio = _ratio(stats.get("abs_max"), base_stats.get("abs_max"))
                lines.append(
                    f"| {layer} | {r['name']} | {abs_mean_ratio or 0:.3f} | "
                    f"{hf_ratio or 0:.3f} | {abs_max_ratio or 0:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- If `body_blocks.*` is close to t00 but `decoder_blocks/dec_out` explodes, the failure is in decoder/head compensation.",
            "- If `skip_fusion` explodes, style pressure is entering through the skip merge.",
            "- If `raw_color` explodes while earlier layers stay bounded, the color head is absorbing the constraint mismatch.",
            "- If `raw_warp` explodes but effective warp remains bounded, the model is pushing against the tanh/warp cap.",
            "",
        ]
    )
    (out_dir / "layer_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-level activation diagnostics for tangent/color-guard models.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "exp/layer_diagnostics/color_guard")
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "style_data/overfit50")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-model", type=str, default="sd15")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (
            "t00",
            ROOT / "exp/diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/epoch_0008.pt",
            ROOT / "exp/diffeomorphic_tangent_sweep/t00_ws0p03_g6_nl0/full_eval/epoch_0008",
        ),
        (
            "cg01",
            ROOT / "exp/color_guard_sweep/cg01_t00_color_lp3/epoch_0008.pt",
            ROOT / "exp/color_guard_sweep/cg01_t00_color_lp3/full_eval/epoch_0008",
        ),
        (
            "cg02",
            ROOT / "exp/color_guard_sweep/cg02_t00_color_lp3_edge1p5/epoch_0008.pt",
            ROOT / "exp/color_guard_sweep/cg02_t00_color_lp3_edge1p5/full_eval/epoch_0008",
        ),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ckpt, eval_dir in specs:
        print(f"[trace] {name}", flush=True)
        results.append(
            run_one(
                name=name,
                checkpoint=ckpt,
                eval_dir=eval_dir,
                image_root=args.image_root,
                out_dir=args.out_dir,
                max_rows=args.max_rows,
                batch_size=args.batch_size,
                image_size=args.image_size,
                device=args.device,
                vae_model=args.vae_model,
                cache_dir=args.cache_dir,
            )
        )
    build_report(results, args.out_dir)
    print(f"Saved layer diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
