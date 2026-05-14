"""Anti-hack visual guard metrics for protocol-750 outputs.

These are lightweight, no-download diagnostics intended to catch cases where
CLIP-content stays high while images are visually noisy or structurally weak.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
OVERFIT50 = WORKSPACE_ROOT / "style_data" / "overfit50"
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def parse_name(path: Path) -> tuple[str, str, str] | None:
    if "_to_" not in path.stem:
        return None
    prefix, target = path.stem.rsplit("_to_", 1)
    if "_" not in prefix:
        return None
    src_style, stem = prefix.split("_", 1)
    return src_style, stem, target


def load_rgb(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img).astype("float32") / 255.0


def load_pil(path: Path, size: int = 256) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


def gray_uint8(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((rgb * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    return gray


def canny_edges(gray: np.ndarray) -> np.ndarray:
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med + 1))
    return cv2.Canny(gray, lo, hi) > 0


def edge_scores(gen_gray: np.ndarray, src_gray: np.ndarray) -> tuple[float, float]:
    ge = canny_edges(gen_gray)
    se = canny_edges(src_gray)
    inter = np.logical_and(ge, se).sum()
    gsum = ge.sum()
    ssum = se.sum()
    union = np.logical_or(ge, se).sum()
    f1 = (2.0 * inter / max(gsum + ssum, 1)).item()
    iou = (inter / max(union, 1)).item()
    return float(f1), float(iou)


def edge_breakdown(gen_gray: np.ndarray, src_gray: np.ndarray) -> tuple[float, float, float, float]:
    ge = canny_edges(gen_gray)
    se = canny_edges(src_gray)
    inter = np.logical_and(ge, se).sum()
    gsum = ge.sum()
    ssum = se.sum()
    precision = (inter / max(gsum, 1)).item()
    recall = (inter / max(ssum, 1)).item()
    dilated = cv2.dilate(se.astype("uint8"), np.ones((5, 5), np.uint8), iterations=1) > 0
    extra = np.logical_and(ge, np.logical_not(dilated)).sum()
    extra_rate = (extra / max(gsum, 1)).item()
    density_ratio = (gsum / max(ssum, 1)).item()
    return float(precision), float(recall), float(extra_rate), float(density_ratio)


def high_freq_energy(rgb: np.ndarray) -> float:
    gray = gray_uint8(rgb).astype("float32") / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap)))


def total_variation(rgb: np.ndarray) -> float:
    return float(np.abs(np.diff(rgb, axis=0)).mean() + np.abs(np.diff(rgb, axis=1)).mean())


def laplacian_variance(rgb: np.ndarray) -> float:
    gray = gray_uint8(rgb).astype("float32") / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


def denoise_residual(rgb: np.ndarray) -> float:
    img = (rgb * 255).astype("uint8")
    smooth = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=7).astype("float32") / 255.0
    return float(np.mean(np.abs(rgb - smooth)))


def blockiness(gray: np.ndarray) -> float:
    arr = gray.astype("float32") / 255.0
    if arr.shape[0] < 16 or arr.shape[1] < 16:
        return 0.0
    vertical_boundary = np.abs(arr[:, 8::8] - arr[:, 7:-1:8]).mean() if arr.shape[1] > 8 else 0.0
    horizontal_boundary = np.abs(arr[8::8, :] - arr[7:-1:8, :]).mean() if arr.shape[0] > 8 else 0.0
    vertical_inner = np.abs(arr[:, 1::8] - arr[:, :-1:8]).mean() if arr.shape[1] > 8 else 1e-8
    horizontal_inner = np.abs(arr[1::8, :] - arr[:-1:8, :]).mean() if arr.shape[0] > 8 else 1e-8
    boundary = float(vertical_boundary + horizontal_boundary)
    inner = float(vertical_inner + horizontal_inner)
    return boundary / max(inner, 1e-8)


def gaussian_blur_pil(pil: Image.Image, radius: float = 1.2) -> Image.Image:
    arr = np.asarray(pil).astype("float32") / 255.0
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=radius, sigmaY=radius)
    return Image.fromarray(np.clip(blur * 255.0, 0, 255).astype("uint8"))


def down_up_pil(pil: Image.Image, scale: float = 0.5) -> Image.Image:
    w, h = pil.size
    down = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return down.resize((w, h), Image.BICUBIC)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype("float32"), cv2.COLOR_RGB2LAB)


def highpass_channel(channel: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(channel.astype("float32"), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return channel.astype("float32") - blur


def chroma_speckle_score(rgb: np.ndarray) -> float:
    lab = rgb_to_lab(rgb)
    l_hp = highpass_channel(lab[:, :, 0])
    a_hp = highpass_channel(lab[:, :, 1])
    b_hp = highpass_channel(lab[:, :, 2])
    chroma = float(np.mean(np.abs(a_hp)) + np.mean(np.abs(b_hp)))
    luma = float(np.mean(np.abs(l_hp)))
    return chroma / max(luma, 1e-8)


def flat_chroma_hf_score(rgb: np.ndarray) -> float:
    lab = rgb_to_lab(rgb)
    l = lab[:, :, 0]
    a_hp = highpass_channel(lab[:, :, 1])
    b_hp = highpass_channel(lab[:, :, 2])
    l_blur = cv2.GaussianBlur(l.astype("float32"), (0, 0), sigmaX=2.0, sigmaY=2.0)
    gx = cv2.Sobel(l_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_blur, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    thresh = np.percentile(edge_mag, 60)
    flat_mask = edge_mag <= thresh
    chroma_hf = np.abs(a_hp) + np.abs(b_hp)
    return float(chroma_hf[flat_mask].mean()) if np.any(flat_mask) else float(chroma_hf.mean())


def get_clip_feat(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]


def encode_clip_pils(clip_model, clip_processor, pils: list[Image.Image], device: torch.device) -> torch.Tensor:
    clip_in = clip_processor(images=pils, return_tensors="pt")
    clip_in = {k: v.to(device) for k, v in clip_in.items()}
    with torch.no_grad():
        feat = get_clip_feat(clip_model.get_image_features(**clip_in)).float()
    return F.normalize(feat, dim=-1)


def build_style_prototypes(
    clip_model,
    clip_processor,
    device: torch.device,
    max_ref_cache: int,
) -> dict[str, torch.Tensor]:
    prototypes = {}
    for target in STYLES:
        style_paths = sorted((OVERFIT50 / target).glob("*.jpg"))
        if max_ref_cache > 0:
            style_paths = style_paths[:max_ref_cache]
        if not style_paths:
            continue
        feats = []
        for start in range(0, len(style_paths), 64):
            batch = [load_pil(p) for p in style_paths[start:start + 64]]
            feats.append(encode_clip_pils(clip_model, clip_processor, batch, device).detach())
        stacked = torch.cat(feats, dim=0)
        proto = stacked.mean(dim=0, keepdim=True)
        prototypes[target] = F.normalize(proto, dim=-1)
    return prototypes


def z_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-6)


def style_ref_stats(target: str) -> dict[str, float]:
    vals = {
        "hf": [],
        "tv": [],
        "residual": [],
        "lap_var": [],
        "chroma_speckle": [],
        "flat_chroma_hf": [],
    }
    for path in sorted((OVERFIT50 / target).glob("*.jpg")):
        rgb = load_rgb(path)
        vals["hf"].append(high_freq_energy(rgb))
        vals["tv"].append(total_variation(rgb))
        vals["residual"].append(denoise_residual(rgb))
        vals["lap_var"].append(laplacian_variance(rgb))
        vals["chroma_speckle"].append(chroma_speckle_score(rgb))
        vals["flat_chroma_hf"].append(flat_chroma_hf_score(rgb))
    stats: dict[str, float] = {}
    for key, items in vals.items():
        stats[key] = float(np.mean(items)) if items else 0.0
        mean, std = z_stats(items)
        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = std
    return stats


def eval_guard(images_dir: Path, max_ref_cache: int = 256) -> dict[str, object]:
    rows = []
    style_noise = {target: style_ref_stats(target) for target in STYLES}
    device = torch.device(DEVICE)
    cache_dir = WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    clip_src = str(cache_dir) if cache_dir.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device, dtype=DTYPE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)
    style_prototypes = build_style_prototypes(clip_model, clip_processor, device, max_ref_cache)
    for gen_path in sorted(images_dir.glob("*.jpg")):
        parsed = parse_name(gen_path)
        if parsed is None:
            continue
        src_style, stem, target = parsed
        if target not in STYLES:
            continue
        src_path = OVERFIT50 / src_style / f"{stem}.jpg"
        if not src_path.exists():
            continue
        gen = load_rgb(gen_path)
        src = load_rgb(src_path)
        gen_pil = load_pil(gen_path)
        gen_gray = gray_uint8(gen)
        src_gray = gray_uint8(src)
        ssim_y = float(structural_similarity(src_gray, gen_gray, data_range=255))
        edge_f1, edge_iou = edge_scores(gen_gray, src_gray)
        edge_precision, edge_recall, extra_edge_rate, edge_density_ratio = edge_breakdown(gen_gray, src_gray)
        src_hf = high_freq_energy(src)
        gen_hf = high_freq_energy(gen)
        src_tv = total_variation(src)
        gen_tv = total_variation(gen)
        src_res = denoise_residual(src)
        gen_res = denoise_residual(gen)
        gen_lap_var = laplacian_variance(gen)
        src_lap_var = laplacian_variance(src)
        gen_block = blockiness(gen_gray)
        edge_density = float(canny_edges(gen_gray).mean())
        chroma_speckle = chroma_speckle_score(gen)
        flat_chroma_hf = flat_chroma_hf_score(gen)
        style_feat = style_prototypes.get(target)
        raw_style_clip = 0.0
        blur_style_clip = 0.0
        down_style_clip = 0.0
        blur_style_drop = 0.0
        down_style_drop = 0.0
        if style_feat is not None:
            gen_feat = encode_clip_pils(clip_model, clip_processor, [gen_pil], device)
            blur_feat = encode_clip_pils(clip_model, clip_processor, [gaussian_blur_pil(gen_pil)], device)
            down_feat = encode_clip_pils(clip_model, clip_processor, [down_up_pil(gen_pil)], device)
            raw_style_clip = float((gen_feat @ style_feat.T).item())
            blur_style_clip = float((blur_feat @ style_feat.T).item())
            down_style_clip = float((down_feat @ style_feat.T).item())
            blur_style_drop = raw_style_clip - blur_style_clip
            down_style_drop = raw_style_clip - down_style_clip
        style_ref = style_noise.get(target, {})
        rows.append(
            {
                "src_style": src_style,
                "target": target,
                "image": gen_path.name,
                "ssim_y": ssim_y,
                "edge_f1": edge_f1,
                "edge_iou": edge_iou,
                "edge_precision": edge_precision,
                "edge_recall": edge_recall,
                "extra_edge_rate": extra_edge_rate,
                "edge_density_ratio": edge_density_ratio,
                "hf_gen": gen_hf,
                "hf_src": src_hf,
                "hf_ratio_src": gen_hf / max(src_hf, 1e-8),
                "hf_ratio_style": gen_hf / max(float(style_ref.get("hf", 0.0)), 1e-8),
                "tv_gen": gen_tv,
                "tv_ratio_src": gen_tv / max(src_tv, 1e-8),
                "tv_ratio_style": gen_tv / max(float(style_ref.get("tv", 0.0)), 1e-8),
                "noise_residual": gen_res,
                "noise_ratio_src": gen_res / max(src_res, 1e-8),
                "noise_ratio_style": gen_res / max(float(style_ref.get("residual", 0.0)), 1e-8),
                "lap_var": gen_lap_var,
                "lap_var_ratio_src": gen_lap_var / max(src_lap_var, 1e-8),
                "lap_var_ratio_style": gen_lap_var / max(float(style_ref.get("lap_var", 0.0)), 1e-8),
                "edge_density": edge_density,
                "blockiness": gen_block,
                "raw_style_clip": raw_style_clip,
                "blur_style_clip": blur_style_clip,
                "down_style_clip": down_style_clip,
                "blur_style_drop": blur_style_drop,
                "down_style_drop": down_style_drop,
                "chroma_speckle": chroma_speckle,
                "chroma_speckle_z": (chroma_speckle - float(style_ref.get("chroma_speckle_mean", 0.0))) / max(float(style_ref.get("chroma_speckle_std", 1.0)), 1e-6),
                "flat_chroma_hf": flat_chroma_hf,
                "flat_chroma_hf_z": (flat_chroma_hf - float(style_ref.get("flat_chroma_hf_mean", 0.0))) / max(float(style_ref.get("flat_chroma_hf_std", 1.0)), 1e-6),
            }
        )

    def avg(items: list[dict[str, object]], key: str) -> float:
        vals = [float(x[key]) for x in items]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    per_target = []
    for target in STYLES:
        items = [r for r in rows if r["target"] == target]
        if not items:
            continue
        per_target.append(
            {
                "target": target,
                "images": len(items),
                "ssim_y": avg(items, "ssim_y"),
                "edge_f1": avg(items, "edge_f1"),
                "edge_iou": avg(items, "edge_iou"),
                "edge_precision": avg(items, "edge_precision"),
                "edge_recall": avg(items, "edge_recall"),
                "extra_edge_rate": avg(items, "extra_edge_rate"),
                "edge_density_ratio": avg(items, "edge_density_ratio"),
                "hf_ratio_src": avg(items, "hf_ratio_src"),
                "hf_ratio_style": avg(items, "hf_ratio_style"),
                "tv_ratio_src": avg(items, "tv_ratio_src"),
                "tv_ratio_style": avg(items, "tv_ratio_style"),
                "noise_residual": avg(items, "noise_residual"),
                "noise_ratio_src": avg(items, "noise_ratio_src"),
                "noise_ratio_style": avg(items, "noise_ratio_style"),
                "lap_var_ratio_src": avg(items, "lap_var_ratio_src"),
                "lap_var_ratio_style": avg(items, "lap_var_ratio_style"),
                "edge_density": avg(items, "edge_density"),
                "blockiness": avg(items, "blockiness"),
                "hf_gen": avg(items, "hf_gen"),
                "raw_style_clip": avg(items, "raw_style_clip"),
                "blur_style_clip": avg(items, "blur_style_clip"),
                "down_style_clip": avg(items, "down_style_clip"),
                "blur_style_drop": avg(items, "blur_style_drop"),
                "down_style_drop": avg(items, "down_style_drop"),
                "chroma_speckle": avg(items, "chroma_speckle"),
                "chroma_speckle_z": avg(items, "chroma_speckle_z"),
                "flat_chroma_hf": avg(items, "flat_chroma_hf"),
                "flat_chroma_hf_z": avg(items, "flat_chroma_hf_z"),
            }
        )

    overall = {
        "target": "ALL",
        "images": len(rows),
        "ssim_y": avg(rows, "ssim_y"),
        "edge_f1": avg(rows, "edge_f1"),
        "edge_iou": avg(rows, "edge_iou"),
        "edge_precision": avg(rows, "edge_precision"),
        "edge_recall": avg(rows, "edge_recall"),
        "extra_edge_rate": avg(rows, "extra_edge_rate"),
        "edge_density_ratio": avg(rows, "edge_density_ratio"),
        "hf_ratio_src": avg(rows, "hf_ratio_src"),
        "hf_ratio_style": avg(rows, "hf_ratio_style"),
        "tv_ratio_src": avg(rows, "tv_ratio_src"),
        "tv_ratio_style": avg(rows, "tv_ratio_style"),
        "noise_residual": avg(rows, "noise_residual"),
        "noise_ratio_src": avg(rows, "noise_ratio_src"),
        "noise_ratio_style": avg(rows, "noise_ratio_style"),
        "lap_var_ratio_src": avg(rows, "lap_var_ratio_src"),
        "lap_var_ratio_style": avg(rows, "lap_var_ratio_style"),
        "edge_density": avg(rows, "edge_density"),
        "blockiness": avg(rows, "blockiness"),
        "hf_gen": avg(rows, "hf_gen"),
        "raw_style_clip": avg(rows, "raw_style_clip"),
        "blur_style_clip": avg(rows, "blur_style_clip"),
        "down_style_clip": avg(rows, "down_style_clip"),
        "blur_style_drop": avg(rows, "blur_style_drop"),
        "down_style_drop": avg(rows, "down_style_drop"),
        "chroma_speckle": avg(rows, "chroma_speckle"),
        "chroma_speckle_z": avg(rows, "chroma_speckle_z"),
        "flat_chroma_hf": avg(rows, "flat_chroma_hf"),
        "flat_chroma_hf_z": avg(rows, "flat_chroma_hf_z"),
    }
    return {"results": per_target + [overall], "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_ref_cache", type=int, default=256)
    args = parser.parse_args()

    result = eval_guard(args.images_dir.resolve(), max_ref_cache=args.max_ref_cache)
    output = args.output or args.images_dir.parent / "eval_guard750.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": result["results"]}, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        keys = [
            "target", "images", "ssim_y", "edge_f1", "edge_iou",
            "edge_precision", "edge_recall", "extra_edge_rate", "edge_density_ratio",
            "hf_ratio_src", "hf_ratio_style", "tv_ratio_src", "tv_ratio_style",
            "noise_residual", "noise_ratio_src", "noise_ratio_style",
            "lap_var_ratio_src", "lap_var_ratio_style", "edge_density", "blockiness", "hf_gen",
            "raw_style_clip", "blur_style_clip", "down_style_clip", "blur_style_drop", "down_style_drop",
            "chroma_speckle", "chroma_speckle_z", "flat_chroma_hf", "flat_chroma_hf_z",
        ]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result["results"])

    print(output)
    print(csv_path)
    print(json.dumps(result["results"][-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
