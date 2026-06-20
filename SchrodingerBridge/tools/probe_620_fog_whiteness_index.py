"""
620 Whitening/Fog Index (WFI) - 输出图片白化/雾化量化指标

指标组成:
1. contrast_ratio: 图像 std / mean (对比度相对亮度的比值)
2. dynamic_range: (p95 - p5) / (p95 + p5) 归一化动态范围
3. saturation_mean: HSV 饱和度通道均值 (越高越不雾)
4. edge_energy: Soblet 边缘能量 (越高结构越清晰)
5. luminance_std: 亮度通道标准差 (越低越雾)
6. wfi_score: 综合白化指数 (0=正常, 1=严重白化)

用法:
    python fog_whiteness_index.py --images_dir <dir> [--source_dir <dir>]
    python fog_whiteness_index.py --eval_json <curve_summary.json>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_image_fog_metrics(img: np.ndarray) -> dict:
    """Compute fog/whitening metrics for a single RGB image (H,W,3) in [0,1]."""
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Contrast ratio: std/mean
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    contrast_ratio = std_val / max(mean_val, 1e-8)

    # Dynamic range (normalized)
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = (p95 - p05) / max(p95 + p05, 1e-8)

    # Luminance std
    luminance_std = std_val

    # Saturation (HSV) - vectorized
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = np.where(mx > 0, delta / mx, 0)
    saturation_mean = float(np.mean(sat))

    # Edge energy (simple Soblet approximation)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    h2, w2 = min(dx.shape[0], dy.shape[0]), min(dx.shape[1], dy.shape[1])
    edge_energy = float(np.mean(np.sqrt(dx[:h2, :w2] ** 2 + dy[:h2, :w2] ** 2)))

    # Whitening fog index (composite)
    # Low contrast_ratio -> more fog
    # Low saturation -> more fog
    # Low luminance_std -> more fog
    # Normalize each to [0,1] where 1=healthy, 0=foggy
    cr_norm = min(contrast_ratio / 0.5, 1.0)  # 0.5 is healthy threshold
    sr_norm = min(saturation_mean / 0.4, 1.0)  # 0.4 is healthy threshold
    dr_norm = min(dynamic_range / 0.6, 1.0)    # 0.6 is healthy threshold

    wfi_score = 1.0 - (0.4 * cr_norm + 0.3 * sr_norm + 0.3 * dr_norm)

    return {
        "contrast_ratio": round(contrast_ratio, 4),
        "dynamic_range": round(dynamic_range, 4),
        "saturation_mean": round(saturation_mean, 4),
        "luminance_std": round(luminance_std, 4),
        "edge_energy": round(edge_energy, 6),
        "wfi_score": round(wfi_score, 4),
    }


def compute_pairwise_fog_metrics(source_img: np.ndarray, gen_img: np.ndarray) -> dict:
    """Compare generated image against source to measure fog/whitening degradation."""
    src_metrics = compute_image_fog_metrics(source_img)
    gen_metrics = compute_image_fog_metrics(gen_img)

    gray_src = 0.299 * source_img[:, :, 0] + 0.587 * source_img[:, :, 1] + 0.114 * source_img[:, :, 2]
    gray_gen = 0.299 * gen_img[:, :, 0] + 0.587 * gen_img[:, :, 1] + 0.114 * gen_img[:, :, 2]

    # Contrast retention ratio (gen/src)
    contrast_retention = gen_metrics["contrast_ratio"] / max(src_metrics["contrast_ratio"], 1e-8)

    # Dynamic range retention
    dr_retention = gen_metrics["dynamic_range"] / max(src_metrics["dynamic_range"], 1e-8)

    # Saturation retention
    sat_retention = gen_metrics["saturation_mean"] / max(src_metrics["saturation_mean"], 1e-8)

    # WFI delta (positive = more foggy than source)
    wfi_delta = gen_metrics["wfi_score"] - src_metrics["wfi_score"]

    return {
        "source": src_metrics,
        "generated": gen_metrics,
        "contrast_retention": round(contrast_retention, 4),
        "dr_retention": round(dr_retention, 4),
        "sat_retention": round(sat_retention, 4),
        "wfi_delta": round(wfi_delta, 4),
    }


def evaluate_directory(images_dir: str, source_dir: str | None = None, sample_count: int = 20) -> dict:
    """Evaluate fog/whitening for a directory of generated images."""
    from glob import glob

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
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

            # Try to find matching source image
            if source_dir:
                fname = os.path.basename(img_path)
                # Source images typically have different naming
                # Just compute source stats if we can find it
                pass

            metrics["filename"] = os.path.basename(img_path)
            all_metrics.append(metrics)
        except Exception as e:
            all_metrics.append({"filename": os.path.basename(img_path), "error": str(e)})

    valid = [m for m in all_metrics if "error" not in m]
    if not valid:
        return {"error": "All images failed", "count": len(all_metrics)}

    avg_metrics = {}
    for key in ["contrast_ratio", "dynamic_range", "saturation_mean", "luminance_std", "wfi_score"]:
        vals = [m[key] for m in valid]
        avg_metrics[f"avg_{key}"] = round(float(np.mean(vals)), 4)
        avg_metrics[f"std_{key}"] = round(float(np.std(vals)), 4)

    return {
        "image_count": len(valid),
        "failed_count": len(all_metrics) - len(valid),
        "metrics": avg_metrics,
        "per_image": all_metrics,
    }


def evaluate_epoch(eval_dir: str, source_dir: str | None = None) -> dict:
    """Evaluate fog metrics for an epoch's generated images."""
    images_dir = os.path.join(eval_dir, "images")
    if not os.path.isdir(images_dir):
        return {"error": f"Images directory not found: {images_dir}"}
    return evaluate_directory(images_dir, source_dir)


def main():
    parser = argparse.ArgumentParser(description="620 Whitening/Fog Index (WFI)")
    parser.add_argument("--images_dir", type=str, help="Directory of generated images")
    parser.add_argument("--source_dir", type=str, default=None, help="Source images for pairwise comparison")
    parser.add_argument("--eval_dir", type=str, help="Epoch eval directory (contains images/ subfolder)")
    parser.add_argument("--sample_count", type=int, default=20, help="Number of images to sample")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.eval_dir:
        result = evaluate_epoch(args.eval_dir, args.source_dir)
    elif args.images_dir:
        result = evaluate_directory(args.images_dir, args.source_dir, args.sample_count)
    else:
        parser.print_help()
        return

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
