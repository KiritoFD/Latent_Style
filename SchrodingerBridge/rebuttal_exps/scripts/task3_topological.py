"""Task 3: Rigid Topological Preservation Metrics.

Computes two rigid geometric measures on existing 750 source-generated pairs:
1. Depth topology: MiDaS depth MSE between source and generated images
2. Edge consistency: Canny Edge IoU (sigma=1.5, thresholds [100, 200])

No image regeneration needed. Uses existing results in results/D5-512/.

Methods evaluated:
- IDT (identity): generated = source (upper bound, depth MSE = 0)
- WEAVE: results/D5-512/weave_oriented_e4/
- SaMam: results/D5-512/samam/
- StyleAligned: results/D5-512/stylealigned/ (geometric measure, manifest-independent)

Threshold target: WEAVE MSE_depth strictly lower than TGT, significantly better
than StyleAligned/SaMam.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WEAVE_ROOT = Path(r"g:\GitHub\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

TEST_DIR = WEAVE_ROOT / "data" / "test"
RESULTS_DIR = WEAVE_ROOT / "results" / "D5-512"
HF_CACHE = str(WEAVE_ROOT / "exp" / "eval_cache" / "hf")
OUTPUT_DIR = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\rebuttal_exps\experiments\rebuttal_20260716\task3_topological")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# Method directories (using canonical method names)
METHOD_DIRS = {
    "IDT":          RESULTS_DIR / "identity",
    "WEAVE":        RESULTS_DIR / "weave_oriented_e4",
    "SaMam":        RESULTS_DIR / "samam",
    "StyleAligned": RESULTS_DIR / "stylealigned",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_source_images():
    """Load all source images from data/test/."""
    sources = []
    for style_dir in sorted(TEST_DIR.iterdir()):
        if not style_dir.is_dir():
            continue
        for img_path in sorted(style_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                sources.append({
                    "src_style": style_dir.name,
                    "src_slug": img_path.stem,
                    "src_path": str(img_path),
                })
    print(f"Loaded {len(sources)} source images across {len(set(s['src_style'] for s in sources))} styles")
    return sources


def find_generated_image(method_dir, src_style, src_slug, tgt_style):
    """Find generated image for a (source, target) pair in a method directory.

    Handles multiple naming conventions:
    - <src_slug>__to__<tgt_style>.png  (compact)
    - <src_style>_<src_slug>__to__<tgt_style>.png  (with single-underscore prefix)
    - <src_slug>_to_<tgt_style>.png  (single-underscore 'to')
    - <src_style>_<src_slug>_to_<tgt_style>.png  (WEAVE format)
    - <src_style>__<src_slug>__to__<tgt_style>.png  (SaMam/StyleAligned format)
    """
    candidates = [
        method_dir / f"{src_slug}__to__{tgt_style}.png",
        method_dir / f"{src_style}_{src_slug}__to__{tgt_style}.png",
        method_dir / f"{src_slug}_to_{tgt_style}.png",
        method_dir / f"{src_style}_{src_slug}_to_{tgt_style}.png",
        method_dir / f"{src_style}__{src_slug}__to__{tgt_style}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_midas_model():
    """Load MiDaS depth estimation model via transformers."""
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    # Do NOT force offline mode - allow download if not cached
    model_name = "Intel/dpt-large"
    parts = model_name.split("/")
    repo_dir = Path(HF_CACHE) / "hub" / f"models--{parts[0]}--{parts[1]}"
    snap_root = repo_dir / "snapshots"

    if snap_root.exists():
        revisions = [p for p in snap_root.iterdir() if p.is_dir()]
        if revisions:
            local_path = str(revisions[0])
            print(f"Loading MiDaS from cache: {local_path}")
            model = DPTForDepthEstimation.from_pretrained(local_path).to("cuda").eval()
            processor = DPTImageProcessor.from_pretrained(local_path)
            return model, processor

    # Download from HF
    print(f"MiDaS not cached, downloading {model_name} to {HF_CACHE}...")
    model = DPTForDepthEstimation.from_pretrained(model_name, cache_dir=HF_CACHE).to("cuda").eval()
    processor = DPTImageProcessor.from_pretrained(model_name, cache_dir=HF_CACHE)
    print("MiDaS downloaded and loaded.")
    return model, processor


@torch.inference_mode()
def compute_depth_mse(model, processor, src_path, gen_path):
    """Compute normalized depth MSE between source and generated images."""
    from PIL import Image

    try:
        src_img = Image.open(src_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")

        # Resize to same dimensions for comparison
        target_size = (384, 384)
        src_img = src_img.resize(target_size, Image.BICUBIC)
        gen_img = gen_img.resize(target_size, Image.BICUBIC)

        # Process through MiDaS
        inputs_src = processor(images=src_img, return_tensors="pt").to("cuda")
        inputs_gen = processor(images=gen_img, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out_src = model(**inputs_src)
            out_gen = model(**inputs_gen)


        # Get predicted depth (normalized to [0, 1])
        depth_src = out_src.predicted_depth
        depth_gen = out_gen.predicted_depth

        # Interpolate to same size if needed
        if depth_src.shape != depth_gen.shape:
            depth_gen = torch.nn.functional.interpolate(
                depth_gen.unsqueeze(0).unsqueeze(0),
                size=depth_src.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze()

        # Normalize each to [0, 1]
        depth_src_np = depth_src.cpu().numpy()
        depth_gen_np = depth_gen.cpu().numpy()

        depth_src_np = (depth_src_np - depth_src_np.min()) / (depth_src_np.max() - depth_src_np.min() + 1e-8)
        depth_gen_np = (depth_gen_np - depth_gen_np.min()) / (depth_gen_np.max() - depth_gen_np.min() + 1e-8)

        # MSE
        mse = float(np.mean((depth_src_np - depth_gen_np) ** 2))
        return mse
    except Exception as e:
        print(f"    ERROR computing depth: {e}")
        return None


def compute_canny_edge_iou(src_path, gen_path):
    """Compute Canny Edge IoU between source and generated images.

    Uses Gaussian smoothing sigma=1.5, Canny thresholds [100, 200].
    """
    import cv2

    try:
        src_img = cv2.imread(str(src_path))
        gen_img = cv2.imread(str(gen_path))

        if src_img is None or gen_img is None:
            return None

        # Resize to same dimensions
        target_size = (384, 384)
        src_img = cv2.resize(src_img, target_size, interpolation=cv2.INTER_CUBIC)
        gen_img = cv2.resize(gen_img, target_size, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing with sigma=1.5
        src_blur = cv2.GaussianBlur(src_gray, (0, 0), sigmaX=1.5)
        gen_blur = cv2.GaussianBlur(gen_gray, (0, 0), sigmaX=1.5)

        # Canny edge detection with thresholds [100, 200]
        src_edges = cv2.Canny(src_blur, 100, 200)
        gen_edges = cv2.Canny(gen_blur, 100, 200)

        # Binarize
        src_bin = (src_edges > 0).astype(np.uint8)
        gen_bin = (gen_edges > 0).astype(np.uint8)

        # IoU
        intersection = np.logical_and(src_bin, gen_bin).sum()
        union = np.logical_or(src_bin, gen_bin).sum()

        if union == 0:
            return 1.0  # Both have no edges
        iou = float(intersection / union)
        return iou
    except Exception as e:
        print(f"    ERROR computing edge IoU: {e}")
        return None


def main():
    print("=" * 70)
    print("Task 3: Rigid Topological Preservation Metrics")
    print("  - MiDaS Depth MSE (lower = better content preservation)")
    print("  - Canny Edge IoU (higher = better edge preservation)")
    print("=" * 70)

    # Load source images
    sources = load_source_images()

    # Load MiDaS model
    print("\nLoading MiDaS depth model...")
    model, processor = load_midas_model()
    print("MiDaS loaded successfully.")

    # For each method, compute metrics on all 750 pairs
    all_results = {}

    for method_name, method_dir in METHOD_DIRS.items():
        print(f"\n{'='*50}")
        print(f"Processing method: {method_name}")
        print(f"  Directory: {method_dir}")
        print(f"{'='*50}")

        if not method_dir.exists() and method_name != "IDT":
            print(f"  SKIP: directory does not exist")
            continue

        # Skip if per-pair CSV already exists (resume support)
        existing_csv = OUTPUT_DIR / f"task3_{method_name}_per_pair.csv"
        if existing_csv.exists():
            print(f"  SKIP: per-pair CSV already exists at {existing_csv}")
            # Load existing results into all_results
            import csv as csv_mod
            with open(existing_csv, "r", encoding="utf-8-sig") as f:
                rows = list(csv_mod.DictReader(f))
            depth_mses = [float(r["depth_mse"]) for r in rows if r["depth_mse"]]
            edge_ious = [float(r["edge_iou"]) for r in rows if r["edge_iou"]]
            all_results[method_name] = {
                "method": method_name,
                "pair_count": len(rows),
                "missing_count": 0,
                "depth_mse": {
                    "mean": float(np.mean(depth_mses)) if depth_mses else None,
                    "median": float(np.median(depth_mses)) if depth_mses else None,
                },
                "edge_iou": {
                    "mean": float(np.mean(edge_ious)) if edge_ious else None,
                    "median": float(np.median(edge_ious)) if edge_ious else None,
                },
            }
            continue

        per_pair_results = []
        pair_count = 0
        missing_count = 0
        t0 = time.time()

        for src in sources:
            src_style = src["src_style"]
            src_slug = src["src_slug"]
            src_path = src["src_path"]

            for tgt_style in TARGET_STYLES:
                # For IDT (identity), generated = source
                if method_name == "IDT":
                    gen_path = src_path
                else:
                    gen_path = find_generated_image(method_dir, src_style, src_slug, tgt_style)

                if gen_path is None or not Path(gen_path).exists():
                    missing_count += 1
                    continue

                # Compute depth MSE
                depth_mse = compute_depth_mse(model, processor, src_path, gen_path)

                # Compute Canny Edge IoU
                edge_iou = compute_canny_edge_iou(src_path, gen_path)

                per_pair_results.append({
                    "src_style": src_style,
                    "tgt_style": tgt_style,
                    "src_slug": src_slug,
                    "depth_mse": depth_mse,
                    "edge_iou": edge_iou,
                })
                pair_count += 1

                if pair_count % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {pair_count} pairs ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        print(f"\n  Total: {pair_count} pairs, {missing_count} missing, {elapsed:.0f}s")

        # Aggregate
        depth_mses = [r["depth_mse"] for r in per_pair_results if r["depth_mse"] is not None]
        edge_ious = [r["edge_iou"] for r in per_pair_results if r["edge_iou"] is not None]

        agg = {
            "method": method_name,
            "pair_count": pair_count,
            "missing_count": missing_count,
            "depth_mse": {
                "mean": float(np.mean(depth_mses)) if depth_mses else None,
                "std": float(np.std(depth_mses)) if depth_mses else None,
                "median": float(np.median(depth_mses)) if depth_mses else None,
                "min": float(np.min(depth_mses)) if depth_mses else None,
                "max": float(np.max(depth_mses)) if depth_mses else None,
            },
            "edge_iou": {
                "mean": float(np.mean(edge_ious)) if edge_ious else None,
                "std": float(np.std(edge_ious)) if edge_ious else None,
                "median": float(np.median(edge_ious)) if edge_ious else None,
                "min": float(np.min(edge_ious)) if edge_ious else None,
                "max": float(np.max(edge_ious)) if edge_ious else None,
            },
        }

        # Per-style breakdown
        per_style = {}
        for style in TARGET_STYLES:
            style_depth = [r["depth_mse"] for r in per_pair_results
                          if r["tgt_style"] == style and r["depth_mse"] is not None]
            style_iou = [r["edge_iou"] for r in per_pair_results
                        if r["tgt_style"] == style and r["edge_iou"] is not None]
            per_style[style] = {
                "depth_mse_mean": float(np.mean(style_depth)) if style_depth else None,
                "edge_iou_mean": float(np.mean(style_iou)) if style_iou else None,
            }

        agg["per_style"] = per_style
        all_results[method_name] = agg

        print(f"\n  Depth MSE: mean={agg['depth_mse']['mean']}, median={agg['depth_mse']['median']}")
        print(f"  Edge IoU:  mean={agg['edge_iou']['mean']}, median={agg['edge_iou']['median']}")

        # Save per-pair CSV
        csv_path = OUTPUT_DIR / f"task3_{method_name}_per_pair.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["src_style", "tgt_style", "src_slug", "depth_mse", "edge_iou"])
            w.writeheader()
            w.writerows(per_pair_results)
        print(f"  Per-pair CSV: {csv_path}")

    # Save summary JSON
    json_path = OUTPUT_DIR / "task3_summary.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Summary saved: {json_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE: Rigid Topological Preservation")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'Depth MSE':<18} {'Edge IoU':<15} {'Pairs':<8}")
    for method, r in all_results.items():
        dm = r["depth_mse"]["mean"] if r.get("depth_mse") else None
        ei = r["edge_iou"]["mean"] if r.get("edge_iou") else None
        pc = r["pair_count"]
        dm_str = f"{dm:.6f}" if dm is not None else "N/A"
        ei_str = f"{ei:.4f}" if ei is not None else "N/A"
        print(f"  {method:<15} {dm_str:<18} {ei_str:<15} {pc:<8}")

    print("\nTASK3_EXIT=0")


if __name__ == "__main__":
    main()
