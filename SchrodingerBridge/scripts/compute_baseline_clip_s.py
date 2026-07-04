"""Compute CLIP-S (clip_style) for all baseline methods.

CLIP-S = cos(CLIP_image(gen), CLIP_image(ref_style_prototype))
- gen: generated image
- ref_style_prototype: mean CLIP image feature of all reference images in target style
- target style parsed from filename: ..._to_{tgt_style}.png

Usage:
    python compute_baseline_clip_s.py --output /mnt/i/exp_baseline_clip_s.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# All baseline methods with gen_dir paths (from exp_extra_metrics_v2.log)
METHODS = {
    # 256 resolution
    "samst_latent_256": "/mnt/i/exp_samst_latent_eval/step_000001/images",
    "adain_256": "/mnt/i/Github/Latent_Style/exp_baseline_256/adain/step_000001/images",
    "wct_256": "/mnt/i/Github/Latent_Style/exp_baseline_256/wct/step_000001/images",
    "samst_256": "/mnt/i/Github/Latent_Style/exp_baseline_256/samst/step_000001/images",
    "samam_256": "/mnt/i/exp_samam/eval_256/samam_final_20k_256/step_020000/images",
    # 512 resolution
    "adain_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/adain",
    "wct_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/wct_vgg19/images",
    "samst_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samst",
    "samam_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samam",
    "sdedit_str0.35_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdedit_str0.35",
    "sdturbo_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdturbo",
    "styleid_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/styleid",
    "cut_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/cut",
    "seedream_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream/images",
    "identity_512": "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdturbo",  # identity = sdturbo src (content_distance=0.0 maps to identity)
}

# Reference images root
REF_ROOT = "/mnt/i/wikiart_distinct5_samam_512_classview/test"

# CLIP cache
CLIP_CACHE = "/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"


def parse_tgt_style(filename):
    """Parse target style from filename: ..._to_{style}.png"""
    stem = Path(filename).stem
    for style in STYLE_NAMES:
        if f"_to_{style}" in stem:
            return style
    # Try __to__ format
    if "__to__" in stem:
        return stem.rsplit("__to__", 1)[1]
    return None


def collect_images(root, max_images=750):
    files = sorted(list(Path(root).glob("*.png")) + list(Path(root).glob("*.jpg")))
    if max_images > 0 and len(files) > max_images:
        files = files[:max_images]
    return files


def compute_clip_features(images, clip_model, processor, device, dtype, batch_size=16):
    """Compute CLIP image features for a list of images."""
    all_feats = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        imgs = [Image.open(f).convert("RGB") for f in chunk]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    if not all_feats:
        return None
    return torch.cat(all_feats, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-images", type=int, default=750)
    p.add_argument("--clip-cache", default=CLIP_CACHE)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load CLIP
    from transformers import CLIPModel, CLIPProcessor
    clip_src = args.clip_cache if Path(args.clip_cache).exists() else "openai/clip-vit-base-patch32"
    print(f"[INFO] Loading CLIP from: {clip_src}")
    clip_model = CLIPModel.from_pretrained(clip_src).to(device=device, dtype=dtype).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    # Precompute reference style prototypes
    print("[INFO] Computing reference style prototypes...")
    ref_prototypes = {}
    for style in STYLE_NAMES:
        ref_dir = Path(REF_ROOT) / style
        if not ref_dir.exists():
            print(f"[WARN] Reference dir not found: {ref_dir}")
            continue
        ref_files = collect_images(ref_dir, max_images=50)  # use up to 50 ref images per style
        if not ref_files:
            print(f"[WARN] No reference images in: {ref_dir}")
            continue
        ref_feats = compute_clip_features(ref_files, clip_model, clip_processor, device, dtype, args.batch_size)
        if ref_feats is not None:
            proto = ref_feats.mean(dim=0)
            proto = F.normalize(proto, dim=-1)
            ref_prototypes[style] = proto
            print(f"  {style}: {len(ref_files)} ref images, proto norm={proto.norm().item():.4f}")

    # Compute CLIP-S for each method
    results = {}
    for method_name, gen_dir in METHODS.items():
        gen_path = Path(gen_dir)
        if not gen_path.exists():
            print(f"[SKIP] {method_name}: gen_dir not found: {gen_dir}")
            results[method_name] = {"error": "gen_dir not found", "clip_s": None}
            continue

        # Special handling for identity_512: use sdturbo images but these are identity (no transfer)
        # Actually identity should be the source images themselves
        gen_files = collect_images(gen_path, args.max_images)
        print(f"[{method_name}] {len(gen_files)} images in {gen_dir}")
        if not gen_files:
            results[method_name] = {"error": "no images", "clip_s": None}
            continue

        # Compute CLIP features for generated images
        gen_feats = compute_clip_features(gen_files, clip_model, clip_processor, device, dtype, args.batch_size)
        if gen_feats is None:
            results[method_name] = {"error": "no features", "clip_s": None}
            continue

        # Compute CLIP-S for each generated image
        scores = []
        for i, f in enumerate(gen_files):
            tgt_style = parse_tgt_style(f.name)
            if tgt_style is None or tgt_style not in ref_prototypes:
                continue
            proto = ref_prototypes[tgt_style].to(device)
            gen_feat = gen_feats[i].to(device).to(dtype)
            sim = float(F.cosine_similarity(gen_feat.unsqueeze(0), proto.unsqueeze(0)).item())
            scores.append(sim)

        if not scores:
            print(f"  [WARN] No valid CLIP-S scores computed")
            results[method_name] = {"n_images": len(gen_files), "clip_s": None}
            continue

        clip_s = sum(scores) / len(scores)
        results[method_name] = {
            "n_images": len(gen_files),
            "n_scored": len(scores),
            "clip_s": clip_s,
        }
        print(f"  CLIP-S: {clip_s:.4f} ({len(scores)} images)")

        # Save intermediate
        Path(args.output).write_text(json.dumps(results, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
