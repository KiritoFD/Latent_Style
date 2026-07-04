"""
620 WFI (Whitening Fog Index) Benchmark.

Runs WFI evaluation on 620 experiment generated images and Seedream reference
images, splits into identity / style_transfer groups, and outputs a side-by-side
comparison table in JSON and CSV format.

Usage:
    python tools/eval_620_wfi_benchmark.py ^
        --checkpoint-dir /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_xxx/ ^
        --seedream-dir "G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750" ^
        --output-dir results/wfi_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reuse the core WFI computation from the existing probe script
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
PROBE_SCRIPT = TOOLS_DIR / "probe_620_fog_whiteness_index.py"

# We import the function directly rather than exec-ing the file.
sys.path.insert(0, str(TOOLS_DIR))
from probe_620_fog_whiteness_index import (  # noqa: E402
    compute_image_fog_metrics,
    _load_image,
    _summarize_metric_rows,
)

# ---------------------------------------------------------------------------
# Known style names for the distinct5 benchmark
# ---------------------------------------------------------------------------

DISTINCT5_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_style_pair_from_filename(filename: str, style_names: list[str] | None = None) -> tuple[str, str] | None:
    """
    Parse (src_style, tgt_style) from a generated image filename.

    Supports two naming conventions:
    - {src_style}_{artist}_{title}_to_{tgt_style}.png
    - {src_style}__{artist}_{title}__to__{tgt_style}.png

    Returns None if parsing fails.
    """
    if style_names is None:
        style_names = DISTINCT5_STYLES

    stem = Path(filename).stem

    # Try double-underscore format first
    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        if "__" in left:
            src_style = left.split("__", 1)[0]
            if src_style in style_names and tgt_style in style_names:
                return src_style, tgt_style
        return None

    # Single-underscore format: {src_style}_{...}_to_{tgt_style}
    if "_to_" not in stem:
        return None

    left, tgt_style = stem.rsplit("_to_", 1)

    # Prefer longest style name first to avoid prefix ambiguity
    for src_style in sorted(style_names, key=lambda x: len(x), reverse=True):
        prefix = f"{src_style}_"
        if left.startswith(prefix):
            if src_style in style_names and tgt_style in style_names:
                return src_style, tgt_style

    return None


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_latest_epoch_dir(checkpoint_dir: Path) -> Path | None:
    """Find the latest epoch_XXXX directory under full_eval/."""
    full_eval = checkpoint_dir / "full_eval"
    if not full_eval.is_dir():
        # Also check full_eval_transfer
        full_eval = checkpoint_dir / "full_eval_transfer"
    if not full_eval.is_dir():
        return None

    epoch_dirs = sorted(full_eval.glob("epoch_*"))
    if not epoch_dirs:
        return None
    return epoch_dirs[-1]


def find_epoch_dir(checkpoint_dir: Path, epoch: str | None) -> Path | None:
    """Find the epoch directory, either specified or auto-detected."""
    if epoch is not None:
        epoch_name = f"epoch_{epoch.zfill(4)}"
        for subdir in ("full_eval", "full_eval_transfer"):
            candidate = checkpoint_dir / subdir / epoch_name
            if candidate.is_dir():
                return candidate
        return None
    return find_latest_epoch_dir(checkpoint_dir)


def collect_images_from_dir(images_dir: Path) -> list[Path]:
    """Collect all PNG/JPG images from a directory (flat)."""
    files = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files.extend(sorted(images_dir.glob(pat)))
    return files


def collect_seedream_images(seedream_dir: Path) -> list[Path]:
    """
    Collect images from the Seedream reference directory.
    Images may be in style subdirectories or flat.
    """
    files = []
    # Check for style subdirectories first
    for style_dir in sorted(seedream_dir.iterdir()):
        if style_dir.is_dir():
            for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                files.extend(sorted(style_dir.glob(pat)))

    # Also check for flat images at the root level
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files.extend(sorted(seedream_dir.glob(pat)))

    return sorted(set(files))


# ---------------------------------------------------------------------------
# WFI evaluation
# ---------------------------------------------------------------------------

def evaluate_images(image_paths: list[Path], style_names: list[str] | None = None) -> dict:
    """
    Evaluate WFI metrics on a list of images, splitting into
    all / identity / style_transfer groups.
    """
    if style_names is None:
        style_names = DISTINCT5_STYLES

    buckets: dict[str, list[dict]] = {
        "all": [],
        "identity": [],
        "style_transfer": [],
    }

    for img_path in image_paths:
        try:
            img = _load_image(str(img_path))
            metrics = compute_image_fog_metrics(img)
            metrics["filename"] = img_path.name
        except Exception as exc:
            metrics = {"filename": img_path.name, "error": str(exc)}

        # Always add to "all"
        buckets["all"].append(dict(metrics))

        # Parse style pair
        pair = parse_style_pair_from_filename(img_path.name, style_names)
        if pair is not None:
            src_style, tgt_style = pair
            if src_style == tgt_style:
                buckets["identity"].append(dict(metrics))
            else:
                buckets["style_transfer"].append(dict(metrics))
        else:
            # Cannot determine pair; add to neither identity nor style_transfer
            pass

    return {
        name: _summarize_metric_rows(rows)
        for name, rows in buckets.items()
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_comparison_json(
    experiment_results: dict[str, dict],
    seedream_results: dict,
    meta: dict,
) -> dict:
    """Build the side-by-side comparison JSON structure."""
    comparison = {
        "meta": meta,
        "experiments": {},
        "seedream": seedream_results,
    }

    for exp_name, results in experiment_results.items():
        comparison["experiments"][exp_name] = results

    return comparison


def build_csv_rows(comparison: dict) -> list[dict]:
    """Flatten the comparison JSON into rows for CSV output."""
    rows = []

    # Experiment rows
    for exp_name, groups in comparison.get("experiments", {}).items():
        for group_name, group_data in groups.items():
            metrics = group_data.get("metrics", {})
            image_count = group_data.get("image_count", 0)
            for metric_key, metric_val in metrics.items():
                rows.append({
                    "source": exp_name,
                    "group": group_name,
                    "image_count": image_count,
                    "metric": metric_key,
                    "value": metric_val,
                })

    # Seedream rows
    seedream = comparison.get("seedream", {})
    for group_name, group_data in seedream.items():
        metrics = group_data.get("metrics", {})
        image_count = group_data.get("image_count", 0)
        for metric_key, metric_val in metrics.items():
            rows.append({
                "source": "Seedream45",
                "group": group_name,
                "image_count": image_count,
                "metric": metric_key,
                "value": metric_val,
            })

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="620 WFI Benchmark: compare Whitening Fog Index across experiments and Seedream"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more experiment directories (e.g. .../620_spatial_bridge/620_xxx/)",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default=None,
        help="Epoch number (e.g. '8'). Auto-detects latest if omitted.",
    )
    parser.add_argument(
        "--seedream-dir",
        type=str,
        default=os.path.join(
            "G:", os.sep, "GitHub", "Latent_Style", "Related_Works",
            "baseline_pipeline", "results", "seedream45_api",
            "distinct5_512_seedream45_windhub_20260607_repaired750",
        ),
        help="Path to Seedream reference directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON and CSV results (default: <checkpoint-dir>/wfi_benchmark)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve experiment epoch directories
    # ------------------------------------------------------------------
    exp_epochs: dict[str, Path] = {}
    for ckpt_dir_str in args.checkpoint_dir:
        ckpt_dir = Path(ckpt_dir_str)
        if not ckpt_dir.is_dir():
            print(f"[WARN] checkpoint-dir not found, skipping: {ckpt_dir}")
            continue

        epoch_dir = find_epoch_dir(ckpt_dir, args.epoch)
        if epoch_dir is None:
            print(f"[WARN] No epoch dir found in: {ckpt_dir}")
            continue

        images_dir = epoch_dir / "images"
        if not images_dir.is_dir():
            print(f"[WARN] No images/ dir in: {epoch_dir}")
            continue

        exp_name = f"{ckpt_dir.name} ({epoch_dir.parent.name}/{epoch_dir.name})"
        exp_epochs[exp_name] = images_dir
        print(f"[INFO] Experiment: {exp_name} -> {images_dir}")

    if not exp_epochs:
        print("[ERROR] No valid experiment directories found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluate 620 experiments
    # ------------------------------------------------------------------
    experiment_results: dict[str, dict] = {}
    for exp_name, images_dir in exp_epochs.items():
        print(f"\n[INFO] Evaluating WFI for: {exp_name}")
        image_paths = collect_images_from_dir(images_dir)
        print(f"       Found {len(image_paths)} images")
        if not image_paths:
            print(f"       [WARN] No images found, skipping.")
            continue
        results = evaluate_images(image_paths)
        experiment_results[exp_name] = results
        for group_name, group_data in results.items():
            count = group_data.get("image_count", 0)
            wfi = group_data.get("metrics", {}).get("avg_wfi_score", "N/A")
            print(f"       {group_name}: {count} images, avg_wfi_score={wfi}")

    # ------------------------------------------------------------------
    # Evaluate Seedream reference
    # ------------------------------------------------------------------
    seedream_results: dict = {}
    seedream_dir = Path(args.seedream_dir)
    if seedream_dir.is_dir():
        print(f"\n[INFO] Evaluating WFI for Seedream reference: {seedream_dir}")
        seedream_paths = collect_seedream_images(seedream_dir)
        print(f"       Found {len(seedream_paths)} images")
        if seedream_paths:
            seedream_results = evaluate_images(seedream_paths)
            for group_name, group_data in seedream_results.items():
                count = group_data.get("image_count", 0)
                wfi = group_data.get("metrics", {}).get("avg_wfi_score", "N/A")
                print(f"       {group_name}: {count} images, avg_wfi_score={wfi}")
        else:
            print("       [WARN] No Seedream images found.")
    else:
        print(f"\n[WARN] Seedream directory not found: {seedream_dir}")

    # ------------------------------------------------------------------
    # Build comparison and write outputs
    # ------------------------------------------------------------------
    meta = {
        "tool": "eval_620_wfi_benchmark",
        "checkpoint_dirs": [str(p) for p in args.checkpoint_dir],
        "epoch": args.epoch or "auto",
        "seedream_dir": str(seedream_dir),
    }

    comparison = build_comparison_json(experiment_results, seedream_results, meta)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: use the first checkpoint dir
        output_dir = Path(args.checkpoint_dir[0]) / "wfi_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = output_dir / "wfi_benchmark_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[INFO] JSON saved to: {json_path}")

    # Write CSV
    csv_rows = build_csv_rows(comparison)
    csv_path = output_dir / "wfi_benchmark_comparison.csv"
    write_csv(csv_rows, csv_path)
    print(f"[INFO] CSV saved to: {csv_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("WFI Benchmark Summary")
    print("=" * 80)

    metric_keys = [
        "avg_contrast_ratio", "avg_dynamic_range", "avg_saturation_mean",
        "avg_luminance_std", "avg_edge_energy", "avg_wfi_score",
    ]

    for exp_name, groups in experiment_results.items():
        print(f"\n  {exp_name}")
        print(f"  {'Group':<20s} " + " ".join(f"{k:>18s}" for k in metric_keys))
        print(f"  {'-'*20} " + " ".join(f"{'-'*18}" for _ in metric_keys))
        for group_name in ("all", "identity", "style_transfer"):
            m = groups.get(group_name, {}).get("metrics", {})
            vals = []
            for k in metric_keys:
                v = m.get(k)
                vals.append(f"{v:.4f}" if v is not None else "N/A")
            print(f"  {group_name:<20s} " + " ".join(f"{v:>18s}" for v in vals))

    if seedream_results:
        print(f"\n  Seedream45")
        print(f"  {'Group':<20s} " + " ".join(f"{k:>18s}" for k in metric_keys))
        print(f"  {'-'*20} " + " ".join(f"{'-'*18}" for _ in metric_keys))
        for group_name in ("all", "identity", "style_transfer"):
            m = seedream_results.get(group_name, {}).get("metrics", {})
            vals = []
            for k in metric_keys:
                v = m.get(k)
                vals.append(f"{v:.4f}" if v is not None else "N/A")
            print(f"  {group_name:<20s} " + " ".join(f"{v:>18s}" for v in vals))

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
