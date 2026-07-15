from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STYLE_ROOT = WORKSPACE / "style_data" / "overfit50"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


RUNS = [
    ("Ours", "epoch_0007", ROOT / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0007" / "images", STYLE_ROOT),
    ("Ours", "epoch_0008", ROOT / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0008" / "images", STYLE_ROOT),
    ("Ours", "residual_1p25", ROOT / "S-add__K-1_C-0_W-20_Col-0" / "residual_scale_sweep_epoch7" / "residual_1p25" / "images", STYLE_ROOT),
    ("SaMST", "samst_strict", WORKSPACE / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "images", STYLE_ROOT),
]


def load_runs_from_manifest(path: Path) -> list[tuple[str, str, Path, Path]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    runs: list[tuple[str, str, Path, Path]] = []
    for row in rows:
        method = str(row.get("method", "")).strip()
        run = str(row.get("run", "")).strip()
        images_dir_raw = str(row.get("images_dir", "")).strip()
        if not method or not run or not images_dir_raw:
            continue
        source_root_raw = str(row.get("source_root", "")).strip()
        runs.append((method, run, Path(images_dir_raw), Path(source_root_raw) if source_root_raw else STYLE_ROOT))
    return runs


def parse_name(path: Path) -> tuple[str, str, str] | None:
    if "_to_" not in path.stem:
        return None
    prefix, target = path.stem.rsplit("_to_", 1)
    if "_" not in prefix:
        return None
    src_style, src_stem = prefix.split("_", 1)
    return src_style, src_stem, target


def find_source(source_root: Path, src_style: str, stem: str) -> Path | None:
    folder = source_root / src_style
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        prefixed = folder / f"{src_style}__{stem}{ext}"
        if prefixed.exists():
            return prefixed
    hits = list(folder.glob(f"{stem}.*"))
    if hits:
        return hits[0]
    hits = list(folder.glob(f"{src_style}__{stem}.*"))
    return hits[0] if hits else None


def load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    img.close()
    return arr


def rgb_to_y(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def rgb_to_yiq_iq(arr: np.ndarray) -> np.ndarray:
    # Cheap chroma proxy; avoids optional skimage dependency while tracking
    # color residuals separately from luminance structure.
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    return np.stack([i, q], axis=-1)


def sobel_mag_and_angle(gray: np.ndarray, sigma: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(gray, sigma=sigma)
    gx = ndimage.sobel(smooth, axis=1, mode="reflect")
    gy = ndimage.sobel(smooth, axis=0, mode="reflect")
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    return mag, ang


def safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else None


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    if av.std() < 1e-8 or bv.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def image_metrics(src_path: Path, gen_path: Path, size: int) -> dict[str, float]:
    src_rgb = load_rgb(src_path, size)
    gen_rgb = load_rgb(gen_path, size)
    src_y = rgb_to_y(src_rgb)
    gen_y = rgb_to_y(gen_rgb)

    src_grad, src_ang = sobel_mag_and_angle(src_y)
    gen_grad, gen_ang = sobel_mag_and_angle(gen_y)
    src_low_grad, _ = sobel_mag_and_angle(ndimage.gaussian_filter(src_y, sigma=2.0), sigma=1.0)
    gen_low_grad, _ = sobel_mag_and_angle(ndimage.gaussian_filter(gen_y, sigma=2.0), sigma=1.0)

    src_thr = np.percentile(src_grad, 85)
    gen_thr = np.percentile(gen_grad, 85)
    src_edge = src_grad >= max(src_thr, 1e-6)
    gen_edge = gen_grad >= max(gen_thr, 1e-6)
    support = ndimage.binary_dilation(src_edge, iterations=2)
    flat = src_grad <= np.percentile(src_grad, 45)

    gen_energy = float(gen_grad.sum() + 1e-8)
    content_edge_energy_share = float(gen_grad[support].sum() / gen_energy)
    flat_edge_energy_share = float(gen_grad[flat].sum() / gen_energy)
    strong_edge_extra_rate = float(np.logical_and(gen_edge, ~support).sum() / max(float(gen_edge.sum()), 1.0))
    strong_edge_flat_rate = float(np.logical_and(gen_edge, flat).sum() / max(float(gen_edge.sum()), 1.0))

    align_mask = np.logical_and(support, gen_grad > np.percentile(gen_grad, 60))
    if align_mask.any():
        # Orientation is pi-periodic for edges, so use cos(2*dtheta).
        orientation = np.cos(2.0 * (gen_ang[align_mask] - src_ang[align_mask]))
        orientation_consistency = float((orientation + 1.0).mean() / 2.0)
    else:
        orientation_consistency = 0.0

    lowpass_grad_corr = corrcoef(src_low_grad, gen_low_grad)

    gen_chroma = rgb_to_yiq_iq(gen_rgb)
    gen_chroma_low = ndimage.gaussian_filter(gen_chroma, sigma=(1.4, 1.4, 0.0))
    chroma_res = np.sqrt(((gen_chroma - gen_chroma_low) ** 2).sum(axis=-1))
    flat_chroma_residual = float(chroma_res[flat].mean())
    flat_chroma_energy_share = float(chroma_res[flat].sum() / (float(chroma_res.sum()) + 1e-8))

    # Higher is better: reward structural edge concentration and orientation,
    # penalize flat-region edge/chroma pollution.
    content_edge_purity = (
        content_edge_energy_share
        + orientation_consistency
        + max(0.0, lowpass_grad_corr)
        - flat_edge_energy_share
        - strong_edge_extra_rate
        - flat_chroma_energy_share
    ) / 3.0

    return {
        "content_edge_energy_share_up": content_edge_energy_share,
        "flat_edge_energy_share_down": flat_edge_energy_share,
        "strong_edge_extra_rate_down": strong_edge_extra_rate,
        "strong_edge_flat_rate_down": strong_edge_flat_rate,
        "orientation_consistency_up": orientation_consistency,
        "lowpass_grad_corr_up": lowpass_grad_corr,
        "flat_chroma_residual_down": flat_chroma_residual,
        "flat_chroma_energy_share_down": flat_chroma_energy_share,
        "content_edge_purity_up": content_edge_purity,
    }


def evaluate_run(method: str, run: str, images_dir: Path, source_root: Path, size: int) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, float]]] = {}
    all_rows: list[dict[str, float]] = []
    for gen_path in sorted(images_dir.glob("*")):
        if gen_path.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_name(gen_path)
        if parsed is None:
            continue
        src_style, stem, target = parsed
        src_path = find_source(source_root, src_style, stem)
        if src_path is None:
            continue
        metrics = image_metrics(src_path, gen_path, size)
        buckets.setdefault(target, []).append(metrics)
        all_rows.append(metrics)

    def summarize(rows: list[dict[str, float]]) -> dict[str, float | None]:
        keys = list(rows[0].keys()) if rows else []
        return {key: safe_mean([row[key] for row in rows]) for key in keys}

    overall = summarize(all_rows)
    out: dict[str, Any] = {
        "method": method,
        "run": run,
        "images": len(all_rows),
        **overall,
    }
    for target, rows in sorted(buckets.items()):
        summary = summarize(rows)
        for key, value in summary.items():
            out[f"{target}_{key}"] = value
    return out


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        "method",
        "run",
        "images",
        "content_edge_purity_up",
        "content_edge_energy_share_up",
        "flat_edge_energy_share_down",
        "strong_edge_extra_rate_down",
        "orientation_consistency_up",
        "lowpass_grad_corr_up",
        "flat_chroma_energy_share_down",
    ]
    lines = [
        "# Content Edge Purity Metrics",
        "",
        "These diagnostics target SaMST-like failures where semantic layout remains recognizable but output edges are dominated by texture/grain rather than content structure.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "content_edge_purity_metrics.csv")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    runs = load_runs_from_manifest(args.manifest) if args.manifest is not None else RUNS
    for method, run, images_dir, source_root in runs:
        if not images_dir.exists():
            print(f"SKIP missing: {images_dir}")
            continue
        print(f"Evaluating {method}/{run}")
        rows.append(evaluate_run(method, run, images_dir, source_root, args.size))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    front = ["method", "run", "images"]
    fieldnames = front + [key for key in fieldnames if key not in front]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_md(rows, args.output.with_suffix(".md"))
    print(args.output)
    print(args.output.with_suffix(".md"))
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
