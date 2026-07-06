"""Unified evaluation of baseline generated images.

Computes CLIP-S, LPIPS (content), MUSIQ for a directory of generated images.
Supports both Photo2Art-256 (5 styles) and WikiArt-20-distinct5 (5 styles).

Naming convention: {src_style}__{src_stem}__to__{tgt_style}.png
  OR {src_style}_{src_stem}_to_{tgt_style}.jpg

Usage:
  python _eval_unified.py --image-dir PATH --dataset {photo2art256|wiki20distinct5} --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Offline mode (must be before torch/transformers import)
# Note: HF_HUB_OFFLINE=1 breaks diffusers from_pretrained; use local_files_only=True instead.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", r"C:\Users\Administrator\.cache\torch")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Dataset configs ──
DATASETS = {
    "photo2art256": {
        "styles": ["cezanne", "Hayao", "monet", "photo", "vangogh"],
        "test_root": Path(r"I:\datasets\legacy256_overfit50\test"),
        "image_size": 256,
    },
    "wiki20distinct5": {
        "styles": ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"],
        "test_root": Path(r"I:\datasets\wikiarts20_512_test"),
        "image_size": 512,
    },
}


def parse_filename(name: str):
    """Parse {src_style}__{src_stem}__to__{tgt_style}.png or {src_style}_{stem}_to_{tgt_style}.jpg"""
    stem = name.rsplit(".", 1)[0] if "." in name else name

    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        parts = left.split("__", 2)
        if len(parts) >= 3:
            return parts[0], parts[2], tgt_style
        elif len(parts) == 2:
            return parts[0], parts[1], tgt_style
        return None, None, None

    # {src_style}_{id}_to_{tgt_style}
    m = re.match(r"^(.+?)_(.+?)_to_(.+)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return None, None, None


def load_image(path, size=256):
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


# ── CLIP-S ──
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip(device):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    from transformers import CLIPModel, CLIPProcessor
    print("[CLIP] Loading openai/clip-vit-base-patch32 (local_files_only=True)...", flush=True)
    _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device).eval()
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _clip_image_features(model, inputs):
    """Extract image feature tensor from CLIP get_image_features.

    Handles both old (returns Tensor) and new (returns BaseModelOutputWithPooling) APIs.
    """
    out = model.get_image_features(**inputs)
    if isinstance(out, torch.Tensor):
        return out.float()
    # Newer transformers: BaseModelOutputWithPooling
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds.float()
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output.float()
    # Fallback: last_hidden_state mean-pool
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1).float()
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


def compute_clip_s(gen_files, dataset_cfg, device, batch_size=8):
    """CLIP-S: cos(CLIP(gen), CLIP(ref_prototype))."""
    model, processor = get_clip(device)
    styles = dataset_cfg["styles"]
    test_root = dataset_cfg["test_root"]
    img_size = dataset_cfg["image_size"]

    # Build ref prototypes
    ref_features = {}
    for style in styles:
        style_dir = test_root / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found", flush=True)
            continue
        ref_files = sorted(list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png")))[:30]
        if not ref_files:
            continue
        feats = []
        for rf in ref_files:
            img = load_image(rf, img_size)
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                f = _clip_image_features(model, inputs)
                f = F.normalize(f, dim=-1)
            feats.append(f)
        proto = torch.cat(feats).mean(0, keepdim=True)
        ref_features[style] = F.normalize(proto, dim=-1)

    # Process gen files
    clip_s_list = []
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = [load_image(f, img_size) for f in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_feats = _clip_image_features(model, inputs)
            gen_feats = F.normalize(gen_feats, dim=-1)

        for i, f in enumerate(chunk):
            _, _, tgt_style = parse_filename(f.name)
            if tgt_style and tgt_style in ref_features:
                s = float((gen_feats[i] * ref_features[tgt_style]).sum().item())
                clip_s_list.append(s)

    return float(np.mean(clip_s_list)) if clip_s_list else None


# ── LPIPS (content) ──
_LPIPS_MODEL = None


def get_lpips(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    import lpips
    print("[LPIPS] Loading alex...", flush=True)
    _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_MODEL


def compute_lpips_content(gen_files, dataset_cfg, device, batch_size=4):
    """LPIPS: gen vs src (same src image from test set)."""
    lpips_fn = get_lpips(device)
    styles = dataset_cfg["styles"]
    test_root = dataset_cfg["test_root"]
    img_size = dataset_cfg["image_size"]

    # Build src lookup: {src_id: path} per src_style
    src_lookup = {}
    for style in styles:
        style_dir = test_root / style
        if not style_dir.exists():
            continue
        for sf in style_dir.iterdir():
            if sf.is_file() and sf.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                src_lookup[(style, sf.stem)] = sf

    lpips_list = []
    for gf in gen_files:
        parsed = parse_filename(gf.name)
        if parsed[0] is None:
            continue
        src_style, src_stem, _ = parsed
        src_file = src_lookup.get((src_style, src_stem))
        if src_file is None:
            # Try with stem suffix matching
            for (s, stem), path in src_lookup.items():
                if s == src_style and src_stem in stem:
                    src_file = path
                    break
        if src_file is None:
            continue

        gen_img = load_image(gf, img_size)
        src_img = load_image(src_file, img_size)

        gen_t = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        src_t = torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        gen_t = gen_t.unsqueeze(0).to(device)
        src_t = src_t.unsqueeze(0).to(device)

        with torch.no_grad():
            d = lpips_fn(gen_t, src_t).item()
        lpips_list.append(d)

    return float(np.mean(lpips_list)) if lpips_list else None


# ── MUSIQ ──
_MUSIQ_MODEL = None


def get_musiq(device):
    global _MUSIQ_MODEL
    if _MUSIQ_MODEL is not None:
        return _MUSIQ_MODEL
    import pyiqa
    print("[MUSIQ] Loading...", flush=True)
    _MUSIQ_MODEL = pyiqa.create_metric("musiq", device=device).eval()
    return _MUSIQ_MODEL


def compute_musiq(gen_files, dataset_cfg, device, batch_size=4):
    """MUSIQ image quality assessment."""
    musiq = get_musiq(device)
    from torchvision import transforms
    img_size = dataset_cfg["image_size"]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    scores = []
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in chunk], dim=0).to(device)
        with torch.no_grad():
            out = musiq(imgs)
        for v in out:
            scores.append(float(v))

    return float(np.mean(scores)) if scores else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=750)
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--skip-lpips", action="store_true")
    p.add_argument("--skip-musiq", action="store_true")
    args = p.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    device = torch.device(args.device)

    print(f"=== Unified evaluation ===", flush=True)
    print(f"  image_dir: {args.image_dir}", flush=True)
    print(f"  dataset: {args.dataset} (styles={dataset_cfg['styles']})", flush=True)
    print(f"  test_root: {dataset_cfg['test_root']}", flush=True)

    # Collect images
    gen_files = sorted(list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg")))
    if args.max_images > 0 and len(gen_files) > args.max_images:
        gen_files = gen_files[:args.max_images]
    print(f"  Found {len(gen_files)} images", flush=True)

    if not gen_files:
        print("[ERROR] No images found", flush=True)
        args.output.write_text(json.dumps({"error": "no images"}))
        return 1

    result = {
        "n_images": len(gen_files),
        "dataset": args.dataset,
        "image_dir": str(args.image_dir),
        "wall_seconds": 0.0,
    }
    t0 = time.time()

    if not args.skip_clip:
        print("[CLIP-S] Computing...", flush=True)
        clip_s = compute_clip_s(gen_files, dataset_cfg, device)
        result["clip_s"] = clip_s
        print(f"  CLIP-S = {clip_s}", flush=True)

    if not args.skip_lpips:
        print("[LPIPS] Computing...", flush=True)
        lpips_val = compute_lpips_content(gen_files, dataset_cfg, device)
        result["lpips"] = lpips_val
        print(f"  LPIPS = {lpips_val}", flush=True)

    if not args.skip_musiq:
        print("[MUSIQ] Computing...", flush=True)
        musiq_val = compute_musiq(gen_files, dataset_cfg, device)
        result["musiq"] = musiq_val
        print(f"  MUSIQ = {musiq_val}", flush=True)

    result["wall_seconds"] = time.time() - t0

    # Cleanup GPU
    global _CLIP_MODEL, _LPIPS_MODEL, _MUSIQ_MODEL
    _CLIP_MODEL = None
    _LPIPS_MODEL = None
    _MUSIQ_MODEL = None
    torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\n[DONE] Results saved to {args.output}", flush=True)
    print(f"  {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
