"""Stronger artifact diagnostics for protocol-750 outputs.

This pack focuses on the failure mode that standard CLIP/LPIPS/edge metrics
miss: outputs that preserve structure but look grainy, dirty, or spatially
incoherent in chroma.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pyiqa
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
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


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0, 0, 255).astype("uint8")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype("float32"), cv2.COLOR_RGB2LAB)


def highpass_channel(channel: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(channel.astype("float32"), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return channel.astype("float32") - blur


def chroma_highpass_mag(rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(rgb)
    a_hp = highpass_channel(lab[:, :, 1])
    b_hp = highpass_channel(lab[:, :, 2])
    return np.sqrt(a_hp * a_hp + b_hp * b_hp)


def bilateral_denoise(rgb: np.ndarray) -> np.ndarray:
    den = cv2.bilateralFilter(to_uint8(rgb), d=7, sigmaColor=30, sigmaSpace=7)
    return den.astype("float32") / 255.0


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
        proto = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)
        prototypes[target] = F.normalize(proto, dim=-1)
    return prototypes


def radial_profile(power: np.ndarray, bins: int = 32) -> np.ndarray:
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rbin = np.clip((r / r.max() * bins).astype(np.int32), 0, bins - 1)
    out = np.zeros(bins, dtype=np.float64)
    cnt = np.zeros(bins, dtype=np.float64)
    np.add.at(out, rbin, power)
    np.add.at(cnt, rbin, 1.0)
    prof = out / np.maximum(cnt, 1.0)
    prof = np.maximum(prof, 1e-8)
    prof /= prof.sum()
    return prof.astype("float32")


def fft_radial_descriptor(rgb: np.ndarray, bins: int = 32) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
    hp = highpass_channel(gray, sigma=1.2)
    fft = np.fft.fftshift(np.fft.fft2(hp))
    power = np.abs(fft) ** 2
    prof = radial_profile(power, bins=bins)
    radii = np.arange(1, bins + 1, dtype=np.float32)
    slope = np.polyfit(np.log(radii), np.log(prof + 1e-8), 1)[0]
    return prof, float(slope)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p.astype("float64"), 1e-8)
    q = np.maximum(q.astype("float64"), 1e-8)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def chroma_autocorr_length(rgb: np.ndarray, threshold: float = 0.1) -> float:
    mag = chroma_highpass_mag(rgb).astype("float32")
    mag = mag - mag.mean()
    fft = np.fft.fft2(mag)
    ac = np.fft.ifft2(np.abs(fft) ** 2).real
    ac = np.fft.fftshift(ac)
    center = ac[ac.shape[0] // 2, ac.shape[1] // 2]
    if abs(center) < 1e-8:
        return 0.0
    ac = ac / center
    prof = radial_profile(np.maximum(ac, 0.0), bins=32)
    for idx, value in enumerate(prof):
        if value < threshold:
            return float(idx)
    return float(len(prof) - 1)


def chroma_moran_i(rgb: np.ndarray) -> float:
    x = chroma_highpass_mag(rgb).astype("float32")
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom < 1e-8:
        return 0.0
    right = (x[:, :-1] * x[:, 1:]).sum()
    down = (x[:-1, :] * x[1:, :]).sum()
    w = (x[:, :-1].size + x[:-1, :].size)
    n = x.size
    return float((n / max(w, 1)) * ((right + down) / denom))


def small_chroma_blob_ratio(rgb: np.ndarray, percentile: float = 85.0, max_blob: int = 8) -> float:
    mag = chroma_highpass_mag(rgb).astype("float32")
    thresh = np.percentile(mag, percentile)
    mask = (mag > thresh).astype("uint8")
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = float(areas.sum())
    if total <= 0:
        return 0.0
    small = float(areas[areas <= max_blob].sum())
    return small / total


def structure_tensor_coherence(rgb: np.ndarray, sigma: float = 1.2) -> float:
    mag = chroma_highpass_mag(rgb).astype("float32")
    gx = cv2.Sobel(mag, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mag, cv2.CV_32F, 0, 1, ksize=3)
    j11 = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    j22 = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    j12 = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    trace = j11 + j22
    delta = np.sqrt(np.maximum((j11 - j22) ** 2 + 4.0 * (j12 ** 2), 0.0))
    lam1 = 0.5 * (trace + delta)
    lam2 = 0.5 * (trace - delta)
    coh = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-8)
    return float(np.mean(coh))


def z_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-6)


def style_ref_stats(target: str, bins: int = 32) -> dict[str, object]:
    paths = sorted((OVERFIT50 / target).glob("*.jpg"))
    fft_profiles = []
    fft_slopes = []
    acls = []
    morans = []
    blobs = []
    coherences = []
    for path in paths:
        rgb = load_rgb(path)
        prof, slope = fft_radial_descriptor(rgb, bins=bins)
        fft_profiles.append(prof)
        fft_slopes.append(slope)
        acls.append(chroma_autocorr_length(rgb))
        morans.append(chroma_moran_i(rgb))
        blobs.append(small_chroma_blob_ratio(rgb))
        coherences.append(structure_tensor_coherence(rgb))
    fft_mean = np.mean(np.stack(fft_profiles, axis=0), axis=0) if fft_profiles else np.full(bins, 1.0 / bins)
    out: dict[str, object] = {"fft_profile_mean": fft_mean.tolist()}
    for key, values in {
        "fft_slope": fft_slopes,
        "chroma_acl": acls,
        "chroma_moran": morans,
        "small_blob_ratio": blobs,
        "structure_tensor_coherence": coherences,
    }.items():
        mean, std = z_stats(values)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def avg(items: list[dict[str, object]], key: str) -> float:
    vals = [float(x[key]) for x in items]
    return round(float(np.mean(vals)), 4) if vals else 0.0


def eval_artifact_pack(images_dir: Path, max_ref_cache: int = 64) -> dict[str, object]:
    device = torch.device(DEVICE)
    cache_dir = WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    clip_src = str(cache_dir) if cache_dir.exists() else "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_src).to(device, dtype=DTYPE).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)
    style_prototypes = build_style_prototypes(clip_model, clip_processor, device, max_ref_cache)
    style_stats = {target: style_ref_stats(target) for target in STYLES}

    musiq_metric = pyiqa.create_metric("musiq", device=DEVICE)
    maniqa_metric = pyiqa.create_metric("maniqa", device=DEVICE)
    dists_metric = pyiqa.create_metric("dists", device=DEVICE)

    rows = []
    for gen_path in sorted(images_dir.glob("*.jpg")):
        parsed = parse_name(gen_path)
        if parsed is None:
            continue
        src_style, stem, target = parsed
        if target not in STYLES:
            continue
        content_path = OVERFIT50 / src_style / f"{stem}.jpg"
        if not content_path.exists():
            continue

        gen_rgb = load_rgb(gen_path)
        gen_pil = load_pil(gen_path)
        den_rgb = bilateral_denoise(gen_rgb)
        den_pil = Image.fromarray(to_uint8(den_rgb))
        proto = style_prototypes[target]
        style_feat = style_stats[target]

        gen_feat = encode_clip_pils(clip_model, clip_processor, [gen_pil], device)
        den_feat = encode_clip_pils(clip_model, clip_processor, [den_pil], device)
        raw_style = float((gen_feat @ proto.T).item())
        den_style = float((den_feat @ proto.T).item())

        musiq = float(musiq_metric(str(gen_path)).item())
        maniqa = float(maniqa_metric(str(gen_path)).item())
        dists = float(dists_metric(str(gen_path), str(content_path)).item())

        lab_gen = rgb_to_lab(gen_rgb)
        lab_den = rgb_to_lab(den_rgb)
        denoise_chroma_delta = float(np.mean(np.abs(lab_gen[:, :, 1:] - lab_den[:, :, 1:])))

        fft_prof, fft_slope = fft_radial_descriptor(gen_rgb)
        fft_mean = np.array(style_feat["fft_profile_mean"], dtype="float32")
        fft_radial_kl = kl_divergence(fft_prof, fft_mean)
        fft_slope_error = abs(fft_slope - float(style_feat["fft_slope_mean"]))

        chroma_acl = chroma_autocorr_length(gen_rgb)
        chroma_moran = chroma_moran_i(gen_rgb)
        blob_ratio = small_chroma_blob_ratio(gen_rgb)
        coherence = structure_tensor_coherence(gen_rgb)
        acl_z = (chroma_acl - float(style_feat["chroma_acl_mean"])) / max(float(style_feat["chroma_acl_std"]), 1e-6)
        moran_z = (chroma_moran - float(style_feat["chroma_moran_mean"])) / max(float(style_feat["chroma_moran_std"]), 1e-6)
        blob_z = (blob_ratio - float(style_feat["small_blob_ratio_mean"])) / max(float(style_feat["small_blob_ratio_std"]), 1e-6)
        coherence_z = (coherence - float(style_feat["structure_tensor_coherence_mean"])) / max(float(style_feat["structure_tensor_coherence_std"]), 1e-6)
        chroma_grain_index = float(np.mean([blob_z, -acl_z, -coherence_z, fft_slope_error]))

        rows.append(
            {
                "target": target,
                "image": gen_path.name,
                "musiq": musiq,
                "maniqa": maniqa,
                "dists_content": dists,
                "denoise_style_drop": raw_style - den_style,
                "denoise_chroma_delta": denoise_chroma_delta,
                "fft_radial_kl_style": fft_radial_kl,
                "fft_slope_error": fft_slope_error,
                "chroma_acl": chroma_acl,
                "chroma_acl_z": acl_z,
                "chroma_moran": chroma_moran,
                "chroma_moran_z": moran_z,
                "small_blob_ratio": blob_ratio,
                "small_blob_ratio_z": blob_z,
                "structure_tensor_coherence": coherence,
                "structure_tensor_coherence_z": coherence_z,
                "chroma_grain_index": chroma_grain_index,
            }
        )

    per_target = []
    for target in STYLES:
        items = [r for r in rows if r["target"] == target]
        if not items:
            continue
        per_target.append(
            {
                "target": target,
                "images": len(items),
                "musiq": avg(items, "musiq"),
                "maniqa": avg(items, "maniqa"),
                "dists_content": avg(items, "dists_content"),
                "denoise_style_drop": avg(items, "denoise_style_drop"),
                "denoise_chroma_delta": avg(items, "denoise_chroma_delta"),
                "fft_radial_kl_style": avg(items, "fft_radial_kl_style"),
                "fft_slope_error": avg(items, "fft_slope_error"),
                "chroma_acl_z": avg(items, "chroma_acl_z"),
                "chroma_moran_z": avg(items, "chroma_moran_z"),
                "small_blob_ratio_z": avg(items, "small_blob_ratio_z"),
                "structure_tensor_coherence_z": avg(items, "structure_tensor_coherence_z"),
                "chroma_grain_index": avg(items, "chroma_grain_index"),
            }
        )

    overall = {
        "target": "ALL",
        "images": len(rows),
        "musiq": avg(rows, "musiq"),
        "maniqa": avg(rows, "maniqa"),
        "dists_content": avg(rows, "dists_content"),
        "denoise_style_drop": avg(rows, "denoise_style_drop"),
        "denoise_chroma_delta": avg(rows, "denoise_chroma_delta"),
        "fft_radial_kl_style": avg(rows, "fft_radial_kl_style"),
        "fft_slope_error": avg(rows, "fft_slope_error"),
        "chroma_acl_z": avg(rows, "chroma_acl_z"),
        "chroma_moran_z": avg(rows, "chroma_moran_z"),
        "small_blob_ratio_z": avg(rows, "small_blob_ratio_z"),
        "structure_tensor_coherence_z": avg(rows, "structure_tensor_coherence_z"),
        "chroma_grain_index": avg(rows, "chroma_grain_index"),
    }
    return {"results": per_target + [overall], "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_ref_cache", type=int, default=64)
    args = parser.parse_args()

    result = eval_artifact_pack(args.images_dir.resolve(), max_ref_cache=args.max_ref_cache)
    output = args.output or args.images_dir.parent / "eval_artifact_pack750.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": result["results"]}, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    keys = [
        "target", "images", "musiq", "maniqa", "dists_content",
        "denoise_style_drop", "denoise_chroma_delta",
        "fft_radial_kl_style", "fft_slope_error",
        "chroma_acl_z", "chroma_moran_z", "small_blob_ratio_z", "structure_tensor_coherence_z",
        "chroma_grain_index",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result["results"])

    print(output)
    print(csv_path)
    print(json.dumps(result["results"][-1], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
