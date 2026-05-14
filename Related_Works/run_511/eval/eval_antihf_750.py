"""Anti high-frequency hack metrics for protocol-750 outputs.

This complements SB-match CLIP/LPIPS metrics with target-calibrated artifact
diagnostics:
  - Blur/downsample style drop
  - HF-z
  - Chroma-HF-z
  - Flat-region-HF-z
  - HF artifact index
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[2]
OVERFIT50 = WORKSPACE_ROOT / "style_data" / "overfit50"
CLIP_CACHE = WORKSPACE_ROOT / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]


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


def to_pil(rgb: np.ndarray) -> Image.Image:
    arr = np.clip(rgb * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def gaussian_low(rgb: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def down_up(rgb: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def hf_energy(rgb: np.ndarray) -> float:
    high = rgb - gaussian_low(rgb)
    return float(np.mean(high * high) / max(float(np.mean(rgb * rgb)), 1e-8))


def chroma_hf(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype("float32")
    l = lab[:, :, 0] / 255.0
    ab = (lab[:, :, 1:3] - 128.0) / 127.0
    l_high = l - cv2.GaussianBlur(l, (0, 0), 1.5)
    ab_low = cv2.GaussianBlur(ab, (0, 0), 1.5)
    ab_high = ab - ab_low
    ab_energy = float(np.mean(ab_high * ab_high))
    l_energy = float(np.mean(l_high * l_high))
    return ab_energy / max(l_energy, 1e-8)


def flat_hf(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype("uint8"), cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
    gray_low = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sx = cv2.Sobel(gray_low, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_low, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sx * sx + sy * sy)
    thresh = float(np.percentile(edge, 60))
    mask = edge <= thresh
    high = np.abs(rgb - gaussian_low(rgb))
    if mask.sum() == 0:
        return float(high.mean())
    return float(high[mask].mean())


def zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / max(std, 1e-6)


def get_clip_feat(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0, :]


def load_clip(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    src = str(CLIP_CACHE) if CLIP_CACHE.exists() else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(src).to(device).eval()
    processor = CLIPProcessor.from_pretrained(src)
    return model, processor


def encode_pils(model, processor, pils: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=pils, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = get_clip_feat(model.get_image_features(**inputs)).float()
    return F.normalize(feats, dim=-1)


def style_paths(target: str, max_ref: int) -> list[Path]:
    paths = sorted((OVERFIT50 / target).glob("*.jpg"))
    return paths[:max_ref] if max_ref > 0 else paths


def style_prototypes(model, processor, device: torch.device, max_ref: int) -> dict[str, torch.Tensor]:
    protos = {}
    for target in STYLES:
        feats = []
        paths = style_paths(target, max_ref)
        for i in range(0, len(paths), 64):
            pils = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
            feats.append(encode_pils(model, processor, pils, device).detach())
        stacked = torch.cat(feats, dim=0)
        proto = stacked.mean(dim=0, keepdim=True)
        protos[target] = F.normalize(proto, dim=-1)
    return protos


def clip_style(model, processor, proto: torch.Tensor, rgb: np.ndarray, device: torch.device) -> float:
    feat = encode_pils(model, processor, [to_pil(rgb)], device)
    return float((feat @ proto.T).item())


def reference_distribution(model, processor, protos: dict[str, torch.Tensor], device: torch.device, max_ref: int) -> dict[str, dict[str, tuple[float, float]]]:
    out = {}
    for target in STYLES:
        vals: dict[str, list[float]] = {
            "hf": [],
            "chroma_hf": [],
            "flat_hf": [],
            "style_drop_blur": [],
            "style_drop_ds": [],
        }
        proto = protos[target]
        for path in style_paths(target, max_ref):
            rgb = load_rgb(path)
            raw = clip_style(model, processor, proto, rgb, device)
            blur = clip_style(model, processor, proto, gaussian_low(rgb), device)
            ds = clip_style(model, processor, proto, down_up(rgb), device)
            vals["hf"].append(hf_energy(rgb))
            vals["chroma_hf"].append(chroma_hf(rgb))
            vals["flat_hf"].append(flat_hf(rgb))
            vals["style_drop_blur"].append(raw - blur)
            vals["style_drop_ds"].append(raw - ds)
        out[target] = {}
        for key, arr in vals.items():
            a = np.asarray(arr, dtype="float32")
            out[target][key] = (float(a.mean()), float(a.std(ddof=0)))
    return out


def eval_antihf(images_dir: Path, max_ref: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_clip(device)
    protos = style_prototypes(model, processor, device, max_ref)
    ref = reference_distribution(model, processor, protos, device, max_ref)

    rows = []
    for gen_path in sorted(images_dir.glob("*.jpg")):
        parsed = parse_name(gen_path)
        if parsed is None:
            continue
        _src_style, _stem, target = parsed
        if target not in STYLES:
            continue
        rgb = load_rgb(gen_path)
        proto = protos[target]
        style_raw = clip_style(model, processor, proto, rgb, device)
        style_blur = clip_style(model, processor, proto, gaussian_low(rgb), device)
        style_ds = clip_style(model, processor, proto, down_up(rgb), device)

        metrics = {
            "hf": hf_energy(rgb),
            "chroma_hf": chroma_hf(rgb),
            "flat_hf": flat_hf(rgb),
            "style_drop_blur": style_raw - style_blur,
            "style_drop_ds": style_raw - style_ds,
        }
        z = {
            f"{key}_z": zscore(value, ref[target][key][0], ref[target][key][1])
            for key, value in metrics.items()
        }
        z_keys = ["hf_z", "chroma_hf_z", "flat_hf_z", "style_drop_blur_z"]
        index = float(np.mean([z[k] for k in z_keys]))
        index_pos = float(np.mean([max(0.0, z[k]) for k in z_keys]))
        rows.append(
            {
                "image": gen_path.name,
                "target": target,
                "style_raw": style_raw,
                "style_blur": style_blur,
                "style_ds": style_ds,
                **metrics,
                **z,
                "hf_artifact_index": index,
                "hf_artifact_index_pos": index_pos,
            }
        )

    def avg(items: list[dict[str, object]], key: str) -> float:
        vals = [float(x[key]) for x in items]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    keys = [
        "style_raw",
        "style_drop_blur",
        "style_drop_ds",
        "hf_z",
        "chroma_hf_z",
        "flat_hf_z",
        "style_drop_blur_z",
        "style_drop_ds_z",
        "hf_artifact_index",
        "hf_artifact_index_pos",
    ]
    results = []
    for target in STYLES:
        items = [r for r in rows if r["target"] == target]
        if items:
            results.append({"target": target, "images": len(items), **{k: avg(items, k) for k in keys}})
    results.append({"target": "ALL", "images": len(rows), **{k: avg(rows, k) for k in keys}})
    return {"results": results, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_ref_cache", type=int, default=256)
    args = parser.parse_args()

    result = eval_antihf(args.images_dir.resolve(), args.max_ref_cache)
    output = args.output or args.images_dir.parent / "eval_antihf750.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": result["results"]}, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    keys = list(result["results"][-1].keys())
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
