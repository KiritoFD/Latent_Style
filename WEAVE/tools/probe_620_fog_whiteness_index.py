"""
620 Whitening/Fog Index (WFI).

Outputs:
1. contrast_ratio = luminance std / mean
2. dynamic_range = (p95 - p5) / (p95 + p5)
3. saturation_mean = mean HSV saturation
4. edge_energy = simple luminance gradient energy
5. luminance_std = luminance standard deviation
6. wfi_score = composite fog/whitening score (0 healthy, 1 very foggy)

Supports:
- plain image directories
- eval roots containing images/
- metrics packets with metrics.csv, split into all_pairs / identity / style_transfer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def _load_image(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_image_fog_metrics(img: np.ndarray) -> dict[str, float]:
    """Compute fog/whitening metrics for one RGB image in [0, 1]."""
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    contrast_ratio = std_val / max(mean_val, 1e-8)

    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = (p95 - p05) / max(p95 + p05, 1e-8)

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-8), 0.0)
    saturation_mean = float(np.mean(sat))

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    h2 = min(dx.shape[0], dy.shape[0])
    w2 = min(dx.shape[1], dy.shape[1])
    edge_energy = float(np.mean(np.sqrt(dx[:h2, :w2] ** 2 + dy[:h2, :w2] ** 2)))

    cr_norm = min(contrast_ratio / 0.5, 1.0)
    sr_norm = min(saturation_mean / 0.4, 1.0)
    dr_norm = min(dynamic_range / 0.6, 1.0)
    wfi_score = 1.0 - (0.4 * cr_norm + 0.3 * sr_norm + 0.3 * dr_norm)

    return {
        "contrast_ratio": round(contrast_ratio, 4),
        "dynamic_range": round(dynamic_range, 4),
        "saturation_mean": round(saturation_mean, 4),
        "luminance_std": round(std_val, 4),
        "edge_energy": round(edge_energy, 6),
        "wfi_score": round(wfi_score, 4),
    }


def compute_pairwise_fog_metrics(source_img: np.ndarray, gen_img: np.ndarray) -> dict[str, float | dict[str, float]]:
    """Compare generated image against source to measure fog/whitening degradation."""
    src_metrics = compute_image_fog_metrics(source_img)
    gen_metrics = compute_image_fog_metrics(gen_img)

    contrast_retention = gen_metrics["contrast_ratio"] / max(src_metrics["contrast_ratio"], 1e-8)
    dr_retention = gen_metrics["dynamic_range"] / max(src_metrics["dynamic_range"], 1e-8)
    sat_retention = gen_metrics["saturation_mean"] / max(src_metrics["saturation_mean"], 1e-8)
    wfi_delta = gen_metrics["wfi_score"] - src_metrics["wfi_score"]

    return {
        "source": src_metrics,
        "generated": gen_metrics,
        "contrast_retention": round(contrast_retention, 4),
        "dr_retention": round(dr_retention, 4),
        "sat_retention": round(sat_retention, 4),
        "wfi_delta": round(wfi_delta, 4),
    }


def _summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"image_count": 0, "failed_count": 0, "metrics": {}, "per_image": []}
    valid = [row for row in rows if "error" not in row]
    if not valid:
        return {"image_count": 0, "failed_count": len(rows), "metrics": {}, "per_image": rows}

    avg_metrics = {}
    for key in ["contrast_ratio", "dynamic_range", "saturation_mean", "luminance_std", "edge_energy", "wfi_score"]:
        vals = [float(row[key]) for row in valid]
        avg_metrics[f"avg_{key}"] = round(float(np.mean(vals)), 4)
        avg_metrics[f"std_{key}"] = round(float(np.std(vals)), 4)

    return {
        "image_count": len(valid),
        "failed_count": len(rows) - len(valid),
        "metrics": avg_metrics,
        "per_image": rows,
    }


def evaluate_directory(images_dir: str, source_dir: str | None = None, sample_count: int = 20) -> dict:
    """Evaluate fog/whitening for a directory of generated images."""
    from glob import glob

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files: list[str] = []
    for pat in patterns:
        image_files.extend(glob(os.path.join(images_dir, pat)))
    image_files = sorted(image_files)[:sample_count]

    if not image_files:
        return {"error": "No images found", "count": 0}

    all_metrics = []
    for img_path in image_files:
        try:
            gen_img = _load_image(img_path)
            metrics = compute_image_fog_metrics(gen_img)
            if source_dir:
                _ = os.path.basename(img_path)
            metrics["filename"] = os.path.basename(img_path)
            all_metrics.append(metrics)
        except Exception as exc:
            all_metrics.append({"filename": os.path.basename(img_path), "error": str(exc)})
    return _summarize_metric_rows(all_metrics)


def evaluate_epoch(eval_dir: str, source_dir: str | None = None) -> dict:
    """Evaluate fog metrics for an epoch eval directory."""
    images_dir = os.path.join(eval_dir, "images")
    if not os.path.isdir(images_dir):
        return {"error": f"Images directory not found: {images_dir}"}
    return evaluate_directory(images_dir, source_dir)


def evaluate_metrics_packet(metrics_csv: str, images_root: str, sample_count: int = 0) -> dict:
    """
    Evaluate WFI over a metrics packet and split results into:
    - all_pairs
    - identity
    - style_transfer
    """
    csv_path = Path(metrics_csv)
    img_root = Path(images_root)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if sample_count > 0:
        rows = rows[:sample_count]

    buckets: dict[str, list[dict]] = {
        "all_pairs": [],
        "identity": [],
        "style_transfer": [],
    }

    for row in rows:
        src_style = str(row.get("src_style", "")).strip()
        tgt_style = str(row.get("tgt_style", "")).strip()
        gen_rel = str(row.get("gen_image", "")).strip()
        img_path = img_root / gen_rel

        target_buckets = ["all_pairs"]
        if src_style and tgt_style and src_style == tgt_style:
            target_buckets.append("identity")
        elif src_style and tgt_style and src_style != tgt_style:
            target_buckets.append("style_transfer")

        try:
            img = _load_image(str(img_path))
            metrics = compute_image_fog_metrics(img)
            metrics.update(
                {
                    "filename": img_path.name,
                    "gen_image": gen_rel,
                    "src_style": src_style,
                    "tgt_style": tgt_style,
                }
            )
        except Exception as exc:
            metrics = {
                "filename": img_path.name,
                "gen_image": gen_rel,
                "src_style": src_style,
                "tgt_style": tgt_style,
                "error": str(exc),
            }

        for bucket in target_buckets:
            buckets[bucket].append(dict(metrics))

    return {
        "packet_metrics_csv": str(csv_path),
        "images_root": str(img_root),
        "groups": {name: _summarize_metric_rows(bucket_rows) for name, bucket_rows in buckets.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="620 Whitening/Fog Index (WFI)")
    parser.add_argument("--images_dir", type=str, help="Directory of generated images")
    parser.add_argument("--source_dir", type=str, default=None, help="Source images for pairwise comparison")
    parser.add_argument("--eval_dir", type=str, help="Epoch eval directory (contains images/ subfolder)")
    parser.add_argument("--metrics_csv", type=str, help="metrics.csv path for grouped packet evaluation")
    parser.add_argument("--images_root", type=str, help="Image root used together with --metrics_csv")
    parser.add_argument("--sample_count", type=int, default=20, help="Number of images to sample")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.metrics_csv:
        if not args.images_root:
            raise SystemExit("--metrics_csv requires --images_root")
        result = evaluate_metrics_packet(args.metrics_csv, args.images_root, args.sample_count)
    elif args.eval_dir:
        result = evaluate_epoch(args.eval_dir, args.source_dir)
    elif args.images_dir:
        result = evaluate_directory(args.images_dir, args.source_dir, args.sample_count)
    else:
        parser.print_help()
        return

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
