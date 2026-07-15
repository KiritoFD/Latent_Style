#!/usr/bin/env python3
"""
Whitening Fog Index (WFI) — quantitative metrics for detecting whitening/fogging in generated images.

Whitening/fogging manifests as:
  1. Low contrast (everything is mid-gray)
  2. Low dynamic range (narrow histogram)
  3. Low color saturation (washed-out colors)
  4. High mean brightness (everything is light)
  5. Low histogram entropy (pixel values cluster in a narrow band)

Metrics computed per image:
  - contrast_ratio: P95 / P5 of luminance. Range [1, 255]. Lower = more whitened.
  - dynamic_range: std of luminance. Lower = more foggy.
  - saturation_mean: mean HSV saturation [0, 1]. Lower = more desaturated.
  - brightness_mean: mean luminance [0, 1]. Higher = more whitened.
  - hist_entropy: Shannon entropy of luminance histogram [0, 8]. Lower = more compressed.
  - wfi_score: composite [0, 1]. Higher = more whitened/foggy.

Usage:
  from utils.wfi import compute_wfi, compute_wfi_for_directory, wfi_benchmark
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _to_numpy_rgb(image: Any) -> np.ndarray:
    """Convert PIL Image / numpy / tensor to numpy RGB uint8 [H, W, 3]."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif hasattr(image, "cpu"):  # torch tensor
        arr = image.cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:  # PIL Image
        arr = np.array(image.convert("RGB"))
    return arr


def _rgb_to_hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    """Compute HSV saturation for RGB uint8 array. Returns [H, W] float [0, 1]."""
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-8), 0.0)
    return sat


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance (Rec. 709) for RGB uint8. Returns [H, W] float [0, 255]."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _histogram_entropy(lum: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy of luminance histogram."""
    hist, _ = np.histogram(lum.flatten(), bins=bins, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_wfi(image: Any) -> dict[str, float]:
    """
    Compute Whitening Fog Index for a single image.

    Args:
        image: PIL Image, numpy array [H,W,3] uint8 or float [0,1], or torch tensor.

    Returns:
        dict with keys: contrast_ratio, dynamic_range, saturation_mean,
                        brightness_mean, hist_entropy, wfi_score
    """
    rgb = _to_numpy_rgb(image)
    lum = _luminance(rgb)
    sat = _rgb_to_hsv_saturation(rgb)

    # Contrast ratio: P95 / P5 of luminance
    p5, p95 = np.percentile(lum, [5, 95])
    contrast_ratio = float(p95 / max(p5, 1.0))  # avoid div by zero

    # Dynamic range: std of luminance
    dynamic_range = float(lum.std())

    # Saturation mean
    saturation_mean = float(sat.mean())

    # Brightness mean [0, 1]
    brightness_mean = float(lum.mean() / 255.0)

    # Histogram entropy
    hist_entropy = _histogram_entropy(lum)

    # Composite WFI score [0, 1], higher = more whitened
    # Normalize each component to [0, 1] where 1 = maximally whitened
    contrast_norm = 1.0 - min(contrast_ratio / 5.0, 1.0)       # 5+ = good contrast
    range_norm = 1.0 - min(dynamic_range / 60.0, 1.0)           # 60+ = good range
    sat_norm = 1.0 - saturation_mean                             # 0 sat = fully whitened
    bright_norm = max(0.0, (brightness_mean - 0.3) / 0.4)       # >0.3 starts whitening
    entropy_norm = 1.0 - min(hist_entropy / 7.0, 1.0)           # 7+ bits = good entropy

    wfi_score = float(
        0.25 * contrast_norm
        + 0.20 * range_norm
        + 0.20 * sat_norm
        + 0.15 * bright_norm
        + 0.20 * entropy_norm
    )

    return {
        "contrast_ratio": contrast_ratio,
        "dynamic_range": dynamic_range,
        "saturation_mean": saturation_mean,
        "brightness_mean": brightness_mean,
        "hist_entropy": hist_entropy,
        "wfi_score": wfi_score,
    }


def compute_wfi_for_directory(
    image_dir: str | Path,
    pattern: str = "*.png",
    recursive: bool = False,
) -> dict[str, Any]:
    """
    Compute WFI for all images in a directory.

    Returns:
        dict with:
          - per_image: {filename: wfi_metrics}
          - aggregate: mean/std of each metric
          - count: number of images
    """
    image_dir = Path(image_dir)
    if recursive:
        files = sorted(image_dir.rglob(pattern))
    else:
        files = sorted(image_dir.glob(pattern))

    # Also include jpg/jpeg
    for ext in ("*.jpg", "*.jpeg"):
        if recursive:
            files.extend(sorted(image_dir.rglob(ext)))
        else:
            files.extend(sorted(image_dir.glob(ext)))
    files = sorted(set(files))

    per_image: dict[str, dict[str, float]] = {}
    metrics_keys = None

    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            wfi = compute_wfi(img)
            per_image[f.name] = wfi
            if metrics_keys is None:
                metrics_keys = list(wfi.keys())
        except Exception as e:
            print(f"  WARNING: failed to process {f}: {e}", file=sys.stderr)

    if not per_image:
        return {"per_image": {}, "aggregate": {}, "count": 0}

    # Aggregate
    aggregate: dict[str, dict[str, float]] = {}
    for key in metrics_keys:
        values = [v[key] for v in per_image.values()]
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    return {
        "per_image": per_image,
        "aggregate": aggregate,
        "count": len(per_image),
    }


def wfi_benchmark(
    eval_dir: str | Path,
    image_subdir: str = "images",
    output_filename: str = "wfi_benchmark.json",
    source_image_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run WFI benchmark on an evaluation output directory.

    Reads images from eval_dir/image_subdir/, computes WFI for each,
    optionally compares against source images, and writes results to
    eval_dir/output_filename.

    Args:
        eval_dir: Evaluation output directory (contains images/ and summary.json)
        image_subdir: Subdirectory containing generated images
        output_filename: Output JSON filename
        source_image_dir: Optional directory of source images for baseline WFI

    Returns:
        WFI benchmark results dict
    """
    eval_dir = Path(eval_dir)
    images_path = eval_dir / image_subdir

    if not images_path.exists():
        print(f"  WARNING: {images_path} does not exist, skipping WFI benchmark")
        return {"error": f"Image directory not found: {images_path}"}

    print(f"  Computing WFI for {images_path}...")
    gen_result = compute_wfi_for_directory(images_path)

    if gen_result["count"] == 0:
        print(f"  WARNING: no images found in {images_path}")
        return {"error": "No images found", "count": 0}

    # Compute source baseline if provided (recursive because source dirs often have style subdirs)
    source_result = None
    if source_image_dir and Path(source_image_dir).exists():
        print(f"  Computing WFI for source images {source_image_dir}...")
        source_result = compute_wfi_for_directory(source_image_dir, recursive=True)

    # Compute per-pair breakdown by parsing filenames
    # Format: {src_style}_{src_stem}_to_{tgt_style}.png
    pair_breakdown: dict[str, dict[str, Any]] = {}
    for fname, metrics in gen_result["per_image"].items():
        # Parse pair from filename
        stem = Path(fname).stem
        if "_to_" in stem:
            parts = stem.split("_to_")
            if len(parts) == 2:
                src_part = parts[0]
                tgt_part = parts[1]
                # src_part = {src_style}_{src_stem}, tgt_part = {tgt_style}
                # src_style is the first token before first underscore
                src_style = src_part.split("_")[0] if "_" in src_part else src_part
                tgt_style = tgt_part
                pair_key = f"{src_style}->{tgt_style}"
                if pair_key not in pair_breakdown:
                    pair_breakdown[pair_key] = []
                pair_breakdown[pair_key].append(metrics)

    # Aggregate per pair
    pair_aggregate: dict[str, dict[str, float]] = {}
    for pair_key, metrics_list in pair_breakdown.items():
        pair_aggregate[pair_key] = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            pair_aggregate[pair_key][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }

    # Identity vs transfer breakdown
    idt_metrics: list[dict] = []
    transfer_metrics: list[dict] = []
    for fname, metrics in gen_result["per_image"].items():
        stem = Path(fname).stem
        if "_to_" in stem:
            parts = stem.split("_to_")
            if len(parts) == 2:
                src_style = parts[0].split("_")[0] if "_" in parts[0] else parts[0]
                tgt_style = parts[1]
                if src_style == tgt_style:
                    idt_metrics.append(metrics)
                else:
                    transfer_metrics.append(metrics)

    def _agg_list(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        if not metrics_list:
            return {}
        result = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
        return result

    result = {
        "generated_wfi": gen_result["aggregate"],
        "generated_count": gen_result["count"],
        "source_wfi": source_result["aggregate"] if source_result else None,
        "source_count": source_result["count"] if source_result else 0,
        "pair_breakdown": pair_aggregate,
        "identity_wfi": _agg_list(idt_metrics),
        "transfer_wfi": _agg_list(transfer_metrics),
    }

    # Write output
    output_path = eval_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  WFI benchmark written to {output_path}")
    print(f"  Generated WFI score (mean): {gen_result['aggregate']['wfi_score']['mean']:.4f}")
    if source_result and source_result.get('aggregate', {}).get('wfi_score'):
        print(f"  Source WFI score (mean): {source_result['aggregate']['wfi_score']['mean']:.4f}")
        delta = gen_result['aggregate']['wfi_score']['mean'] - source_result['aggregate']['wfi_score']['mean']
        print(f"  WFI delta (gen - source): {delta:+.4f} ({'whitened' if delta > 0 else 'not whitened'})")

    # Also append to summary.json if it exists
    summary_path = eval_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            summary["wfi_benchmark"] = result
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Appended WFI results to {summary_path}")
        except Exception as e:
            print(f"  WARNING: could not append to summary.json: {e}")

    return result


def main():
    """CLI entry point: python -m utils.wfi <eval_dir> [--source-dir <dir>]"""
    import argparse

    parser = argparse.ArgumentParser(description="Whitening Fog Index benchmark")
    parser.add_argument("eval_dir", type=str, help="Evaluation output directory")
    parser.add_argument("--source-dir", type=str, default=None, help="Source images directory for baseline")
    parser.add_argument("--image-subdir", type=str, default="images")
    args = parser.parse_args()

    result = wfi_benchmark(args.eval_dir, args.image_subdir, source_image_dir=args.source_dir)

    # Print summary table
    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    print("\n=== WFI Benchmark Summary ===")
    print(f"Generated images: {result['generated_count']}")
    if result["source_count"]:
        print(f"Source images: {result['source_count']}")

    gen = result["generated_wfi"]
    src = result.get("source_wfi")

    print(f"\n{'Metric':<20} {'Generated':>12} {'Source':>12} {'Delta':>10}")
    print("-" * 56)
    for key in ["wfi_score", "contrast_ratio", "dynamic_range", "saturation_mean", "brightness_mean", "hist_entropy"]:
        g_mean = gen[key]["mean"]
        s_mean = src[key]["mean"] if src else 0.0
        delta = g_mean - s_mean if src else 0.0
        print(f"{key:<20} {g_mean:>12.4f} {s_mean:>12.4f} {delta:>+10.4f}")

    # Transfer vs identity
    if result.get("transfer_wfi") and result.get("identity_wfi"):
        print(f"\n{'Metric':<20} {'Transfer':>12} {'Identity':>12} {'Delta':>10}")
        print("-" * 56)
        for key in ["wfi_score", "contrast_ratio", "dynamic_range", "saturation_mean"]:
            t = result["transfer_wfi"].get(key, {}).get("mean", 0)
            i = result["identity_wfi"].get(key, {}).get("mean", 0)
            print(f"{key:<20} {t:>12.4f} {i:>12.4f} {t-i:>+10.4f}")


if __name__ == "__main__":
    main()
