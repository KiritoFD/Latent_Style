#!/usr/bin/env python3
"""SaMST inference for WikiArt-20 distinct5 subset.

Uses the existing 5-style SaMST checkpoint (epoch_20.model) to generate
stylized images for the 5 distinct wikiart styles.

The SaMST checkpoint was trained on 5 styles. The style_id index (0..5) maps
to specific styles in the training order. This script applies all available
style indices to each content image and saves with proper naming.

Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
Output structure: {output_dir}/*.png  +  {output_dir}/../_DONE

Usage:
  python _gen_samst_wiki20.py \
    --test-dir I:\datasets\wikiarts20_512_test \
    --output-dir I:\...\exp\baseline_wikiarts20\samst\images \
    --checkpoint I:\...\SaMST-main\checkpoint\epoch_20.model \
    --samst-root I:\...\SaMST-main \
    --styles "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e" \
    --max-src-per-style 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to SaMST .model checkpoint")
    p.add_argument("--samst-root", type=Path, required=True,
                   help="Path to SaMST-main repo root (contains networks/ and test_model/)")
    p.add_argument("--styles", type=str, required=True,
                   help="Comma-separated style names (must match checkpoint training order)")
    p.add_argument("--style-num", type=int, default=0,
                   help="Number of styles in checkpoint (0=auto-detect from styles count)")
    p.add_argument("--max-src-per-style", type=int, default=30)
    p.add_argument("--image-size", type=int, default=512)
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles:
        print("[ERROR] No styles provided", flush=True)
        return 1

    # Auto-detect style_num from checkpoint if not specified
    if args.style_num > 0:
        style_num = args.style_num
    else:
        # Load state_dict to count style_para_list entries
        sd = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        style_para_indices = set()
        for k in sd.keys():
            if "style_para_list" in k and k.endswith(".params"):
                idx = int(k.split("style_para_list.")[1].split(".")[0])
                style_para_indices.add(idx)
        # style_para_list has indices 0..N (index 0 is identity)
        # style_num = N (number of styles, excluding identity)
        style_num = max(style_para_indices) if style_para_indices else len(styles)
        print(f"[INFO] Auto-detected style_num={style_num} from checkpoint (indices={sorted(style_para_indices)})", flush=True)
        del sd

    print(f"=== SaMST WikiArt-20 distinct5 inference ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  test_dir: {args.test_dir}", flush=True)
    print(f"  output_dir: {args.output_dir}", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(f"  styles({len(styles)}): {styles}", flush=True)
    print(f"  style_num: {style_num}", flush=True)

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", flush=True)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Add SaMST repo to path
    sys.path.insert(0, str(args.samst_root))
    sys.path.insert(0, str(args.samst_root / "test_model"))

    # Import SaMST modules
    try:
        from networks.transfer_net import TransformerNet
        from test_model import utils as samst_utils
        print("[INFO] Loaded SaMST modules", flush=True)
    except ImportError as e:
        print(f"[ERROR] Failed to import SaMST modules: {e}", flush=True)
        print(f"  sys.path: {sys.path}", flush=True)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    # Load model
    try:
        model = TransformerNet(style_num=style_num)
        state_dict = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        print("[INFO] SaMST model loaded", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load SaMST model: {e}", flush=True)
        return 1

    # Collect source images
    import random
    rng = random.Random(42)
    sources = []
    for style in styles:
        style_dir = args.test_dir / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found, skipping", flush=True)
            continue
        imgs = sorted(p for p in style_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(imgs)
        if args.max_src_per_style > 0:
            imgs = imgs[:args.max_src_per_style]
        for p in imgs:
            sources.append((style, p))

    total = len(sources) * len(styles)
    print(f"  {len(sources)} srcs x {len(styles)} styles = {total} images", flush=True)

    # Count existing
    existing = len(list(args.output_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)
    if existing >= total:
        print("  All images exist, skipping.", flush=True)
        (args.output_dir.parent / "_DONE").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return 0

    # Inference
    content_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ])

    n_new = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()

    with torch.no_grad():
        for src_style, src_path in sources:
            src_stem = src_path.stem
            try:
                content_img = Image.open(src_path).convert("RGB")
                content_tensor = content_transform(content_img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[WARN] Failed to load {src_path}: {e}", flush=True)
                n_fail += 1
                continue

            for tgt_idx, tgt_style in enumerate(styles):
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = args.output_dir / out_name
                if out_path.exists():
                    n_skip += 1
                    continue
                try:
                    # SaMST style_id is 1-based (0=identity, 1..style_num are real styles)
                    # Use tgt_idx+1, clamped to style_num
                    sid = min(tgt_idx + 1, style_num)
                    output, _ = model(content_tensor, style_id=[sid])
                    out_img = output[0].cpu().clamp(0, 255) / 255.0
                    transforms.ToPILImage()(out_img).save(out_path)
                    n_new += 1
                except Exception as e:
                    print(f"[WARN] Failed {src_style}->{tgt_style} ({src_stem}): {e}", flush=True)
                    n_fail += 1

            if (n_new + n_skip) % 50 == 0:
                elapsed = time.time() - t0
                rate = (n_new + n_skip) / max(elapsed, 1)
                eta = (total - n_new - n_skip) / max(rate, 0.01)
                print(f"  progress: {n_new + n_skip}/{total}  new={n_new} skip={n_skip} "
                      f"fail={n_fail}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {n_new} new + {n_skip} skipped + {n_fail} failed in {elapsed:.1f}s", flush=True)

    (args.output_dir.parent / "_DONE").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
