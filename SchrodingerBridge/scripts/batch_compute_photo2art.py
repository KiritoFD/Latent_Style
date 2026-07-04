"""Compute CLIP-S, CLIP-T, MUSIQ, ART-FID (content_distance) for photo2art 5 styles.

Styles: cezanne, Hayao, monet, photo, vangogh
Test set: /mnt/i/legacy256_overfit50/test

Based on batch_compute_extra_metrics.py, adapted for photo2art 5 styles.
Adds CLIP-S computation (cos(CLIP(gen), CLIP(ref_prototype))).

Usage:
    python batch_compute_photo2art.py --methods-json methods.json --output results.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Try local src first, then remote SchrodingerBridge src
for src_path in [PROJECT_ROOT / "src", Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/src")]:
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

# photo2art 5 styles
STYLE_NAMES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
STYLE_PROMPTS = {
    "cezanne": "a painting in Paul Cezanne style",
    "Hayao": "a painting in Hayao Miyazaki style",
    "monet": "a painting in Claude Monet style",
    "photo": "a photograph",
    "vangogh": "a painting in Vincent van Gogh style",
}

TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")


def parse_filename(name: str):
    """Parse generated filename to (src_style, src_stem, tgt_style).

    Handles:
    1. {src_style}_{id}_to_{tgt_style}.jpg (seedream/identity format)
    2. {src_style}__{src_style}__{artist}_{title}__to__{tgt_style}.png
    3. {src_style}__{artist}_{title}_to_{tgt_style}.png (cut format)
    """
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
    m = re.match(r"^(cezanne|Hayao|monet|photo|vangogh)_(.+?)_to_(cezanne|Hayao|monet|photo|vangogh)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Generic _to_ format
    m = re.match(r"^(.+?)__(.+?)_to_(.+)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    return None, None, None


def collect_image_files(root: Path, max_images=750):
    files = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if max_images > 0 and len(files) > max_images:
        files = files[:max_images]
    return files


def compute_ref_prototypes(clip_model, clip_processor, device, dtype, max_per_style=30):
    """Compute CLIP image feature prototypes for each style from test set."""
    ref_features = {}
    for style in STYLE_NAMES:
        style_dir = TEST_ROOT / style
        ref_files = sorted(list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png")))[:max_per_style]
        if not ref_files:
            print(f"[WARN] No ref images for {style}")
            continue
        feats = []
        for rf in ref_files:
            img = Image.open(rf).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                      for k, v in inputs.items()}
            with torch.no_grad():
                f = clip_model.get_image_features(**inputs).float()
                f = F.normalize(f, dim=-1)
            feats.append(f)
        proto = torch.cat(feats).mean(0, keepdim=True)
        ref_features[style] = F.normalize(proto, dim=-1)
    return ref_features


def compute_clip_s_and_t(gen_files, clip_model, clip_processor, ref_prototypes, device, dtype, batch_size=16):
    """CLIP-S: cos(CLIP(gen), CLIP(ref_prototype)).
    CLIP-T: cos(CLIP(gen), CLIP_text(style_prompt))."""
    # Text features
    text_feats = {}
    for style, prompt in STYLE_PROMPTS.items():
        inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            tfeat = clip_model.get_text_features(**inputs)
            tfeat = F.normalize(tfeat, dim=-1)
        text_feats[style] = tfeat

    clip_s_scores = []
    clip_t_scores = []
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        inputs = clip_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            ifeat = clip_model.get_image_features(**inputs)
            ifeat = F.normalize(ifeat, dim=-1)

        for i, f in enumerate(chunk):
            _, _, tgt_style = parse_filename(f.name)
            if tgt_style is None or tgt_style not in text_feats:
                continue
            clip_t = float((ifeat[i] * text_feats[tgt_style]).sum().item())
            clip_t_scores.append(clip_t)
            if tgt_style in ref_prototypes:
                clip_s = float((ifeat[i] * ref_prototypes[tgt_style]).sum().item())
                clip_s_scores.append(clip_s)

    clip_s = sum(clip_s_scores) / len(clip_s_scores) if clip_s_scores else None
    clip_t = sum(clip_t_scores) / len(clip_t_scores) if clip_t_scores else None
    return clip_s, clip_t


def compute_musiq(gen_files, musiq_metric, device, batch_size=8):
    scores = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    for start in range(0, len(gen_files), batch_size):
        chunk = gen_files[start:start + batch_size]
        imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in chunk], dim=0).to(device)
        with torch.no_grad():
            out = musiq_metric(imgs)
        for v in out:
            scores.append(float(v))
    return sum(scores) / len(scores) if scores else None


def compute_artfid(gen_files, ref_root, src_root, device, batch_size=2, max_gen=200, max_ref=200):
    """ART-FID = (1 + FID) * (1 + LPIPS_content)."""
    from utils.artfid_metric import (
        load_artfid_feature_extractor,
        load_artfid_lpips,
        collect_artfid_features_from_paths,
        compute_artfid_fid_from_features,
    )

    ref_files = sorted(list(Path(ref_root).glob("**/*.png")) + list(Path(ref_root).glob("**/*.jpg")))
    if max_ref > 0 and len(ref_files) > max_ref:
        ref_files = ref_files[:max_ref]
    src_files = sorted(list(Path(src_root).glob("**/*.png")) + list(Path(src_root).glob("**/*.jpg")))

    if len(gen_files) < 2 or len(ref_files) < 2:
        return None

    feat_extractor = load_artfid_feature_extractor(device=device)
    gen_feats = collect_artfid_features_from_paths(
        [str(f) for f in gen_files[:max_gen]], model=feat_extractor, device=device, batch_size=batch_size
    )
    ref_feats = collect_artfid_features_from_paths(
        [str(f) for f in ref_files], model=feat_extractor, device=device, batch_size=batch_size
    )
    fid = compute_artfid_fid_from_features(gen_feats, ref_feats=ref_feats)

    lpips_model = load_artfid_lpips(device=device)
    src_lookup = {f.stem: f for f in src_files}
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    content_distances = []
    for gf in gen_files[:max_gen]:
        src_style, src_stem, _ = parse_filename(gf.name)
        if src_stem is None:
            continue
        # Try exact stem match, then style_id match
        src_path = src_lookup.get(src_stem)
        if src_path is None and src_style:
            # Try finding src file by stem in src_style dir
            for s in src_files:
                if src_stem in s.stem and (src_style in str(s) or True):
                    src_path = s
                    break
        if src_path is None:
            continue
        try:
            gen_img = transform(Image.open(gf).convert("RGB")).unsqueeze(0).to(device)
            src_img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                d = lpips_model(gen_img, src_img)
            content_distances.append(float(d))
        except Exception:
            pass

    cd = sum(content_distances) / len(content_distances) if content_distances else None
    art_fid = (1.0 + float(fid)) * (1.0 + float(cd)) if (fid is not None and cd is not None) else None
    return {"art_fid": art_fid, "fid": float(fid) if fid is not None else None, "content_distance": cd}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=750)
    p.add_argument("--max-gen-artfid", type=int, default=200)
    p.add_argument("--clip-cache", default="")
    p.add_argument("--skip-clipt", action="store_true")
    p.add_argument("--skip-musiq", action="store_true")
    p.add_argument("--skip-artfid", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load CLIP
    clip_model = None
    clip_processor = None
    clip_dtype = torch.float16 if device.type == "cuda" else torch.float32
    from transformers import CLIPModel, CLIPProcessor
    clip_cache = args.clip_cache.strip()
    if clip_cache and Path(clip_cache).exists():
        clip_src = str(clip_cache)
    else:
        clip_src = "openai/clip-vit-base-patch32"
    print(f"[INFO] Loading CLIP from: {clip_src}")
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=clip_dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    # Compute ref prototypes for CLIP-S
    print("[INFO] Computing style reference prototypes...")
    ref_prototypes = compute_ref_prototypes(clip_model, clip_processor, device, clip_dtype)

    # Load MUSIQ
    musiq_metric = None
    if not args.skip_musiq:
        import pyiqa
        musiq_metric = pyiqa.create_metric("musiq", device=device)

    methods = json.loads(Path(args.methods_json).read_text())
    results = {}

    for method_name, paths in methods.items():
        gen_dir = Path(paths["gen_dir"])
        ref_dir = paths.get("ref_dir", str(TEST_ROOT))
        src_dir = paths.get("src_dir", str(TEST_ROOT))

        if not gen_dir.exists():
            print(f"[SKIP] {method_name}: gen_dir not found: {gen_dir}")
            results[method_name] = {"error": "gen_dir not found"}
            continue

        gen_files = collect_image_files(gen_dir, args.max_images)
        print(f"\n[{method_name}] {len(gen_files)} images in {gen_dir}")
        if not gen_files:
            results[method_name] = {"error": "no images"}
            continue

        m = {"n_images": len(gen_files)}
        t0 = time.time()

        # CLIP-S + CLIP-T
        try:
            clip_s, clip_t = compute_clip_s_and_t(
                gen_files, clip_model, clip_processor, ref_prototypes, device, clip_dtype
            )
            m["clip_s"] = clip_s
            print(f"  CLIP-S: {clip_s:.4f}" if clip_s else "  CLIP-S: None")
            if not args.skip_clipt:
                m["clip_t"] = clip_t
                print(f"  CLIP-T: {clip_t:.4f}" if clip_t else "  CLIP-T: None")
        except Exception as e:
            m["clip_s"] = None
            print(f"  CLIP ERROR: {e}")

        # MUSIQ
        if musiq_metric is not None:
            try:
                musiq = compute_musiq(gen_files, musiq_metric, device)
                m["musiq"] = musiq
                print(f"  MUSIQ: {musiq:.4f}" if musiq else "  MUSIQ: None")
            except Exception as e:
                m["musiq"] = None
                print(f"  MUSIQ ERROR: {e}")

        # ART-FID (includes content_distance = LPIPS)
        if not args.skip_artfid:
            try:
                artfid = compute_artfid(gen_files, ref_dir, src_dir, device,
                                        max_gen=args.max_gen_artfid, batch_size=2)
                m.update(artfid if artfid else {"art_fid": None})
                print(f"  ART-FID: {artfid}")
            except Exception as e:
                m["art_fid"] = None
                print(f"  ART-FID ERROR: {e}")

        m["wall_seconds"] = time.time() - t0
        results[method_name] = m
        Path(args.output).write_text(json.dumps(results, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
