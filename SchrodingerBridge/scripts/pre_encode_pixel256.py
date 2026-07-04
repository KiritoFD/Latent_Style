#!/usr/bin/env python
r"""Pre-encode 256x256 pixel-space images as .pt tensors for the pixel-256 SFM ablation.

Walks a source directory of images (*.jpg / *.png), resizes each to 256x256 with LANCZOS,
converts to a float32 tensor of shape (3, 256, 256) normalized to [-1, 1], and saves as .pt
so the existing AdaCUTLatentDataset loader (which only accepts .pt/.npy) can consume them
without any code changes.

CPU-only, minimal RAM (one image in memory at a time). Supports resume (skips existing .pt).

Two usage modes:
  1) Single style directory:
     python scripts/pre_encode_pixel256.py \
         --source "F:\wikiart_distinct5_samam_512_classview_real\train\Early_Renaissance" \
         --target "F:\wikiart_distinct5_samam_512_pixel256\train\Early_Renaissance" \
         --max 50

  2) Parent directory with multiple style sub-directories:
     python scripts/pre_encode_pixel256.py \
         --source "F:\wikiart_distinct5_samam_512_classview_real\train" \
         --target "F:\wikiart_distinct5_samam_512_pixel256\train"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 256


def encode_one(img_path: Path) -> torch.Tensor:
    """Load image, resize to 256x256 LANCZOS, return float32 tensor (3, 256, 256) in [-1, 1]."""
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()  # (3, H, W)
    tensor = tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
    return tensor.contiguous()


def process_style(src_dir: Path, tgt_dir: Path, max_images: int) -> int:
    """Encode all images in *src_dir* into *tgt_dir* as .pt files. Returns count written."""
    images = sorted(
        Path(e.path) for e in os.scandir(src_dir)
        if Path(e.name).suffix.lower() in IMAGE_EXTS and e.is_file()
    )
    if max_images > 0:
        images = images[:max_images]

    tgt_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        out_path = tgt_dir / (img_path.stem + ".pt")
        if out_path.exists():
            skipped += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(images)}] skipped (exists) - {img_path.name}", flush=True)
            continue
        try:
            tensor = encode_one(img_path)
            torch.save(tensor, out_path)
            written += 1
        except Exception as exc:
            print(f"  ERROR on {img_path}: {exc}", file=sys.stderr, flush=True)
            continue
        if i % 100 == 0:
            print(f"  [{i}/{len(images)}] wrote {out_path.name} ({time.time()-t0:.1f}s)", flush=True)

    print(
        f"  done: {written} written, {skipped} skipped, {len(images)} total in {src_dir.name} "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-encode 256x256 pixel images as .pt tensors for pixel-256 SFM ablation"
    )
    parser.add_argument("--source", type=str, required=True, help="Source image dir (single style or parent of styles)")
    parser.add_argument("--target", type=str, required=True, help="Target .pt output dir (single style or parent of styles)")
    parser.add_argument("--max", type=int, default=0, help="Max images per style (0 = all). For smoke testing.")
    args = parser.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    max_images = max(0, int(args.max))

    if not src.is_dir():
        print(f"ERROR: source dir does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    # Detect mode: if source contains image files directly -> single style.
    # If source contains sub-directories with images -> multi-style.
    # Use os.scandir (robust against stat failures on some Windows volumes).
    entries = list(os.scandir(src))
    direct_images = [Path(e.path) for e in entries
                     if Path(e.name).suffix.lower() in IMAGE_EXTS and e.is_file()]
    style_subdirs = [Path(e.path) for e in entries
                     if e.is_dir() and not e.name.startswith(".") and not e.name.startswith(".latent")]

    if direct_images:
        # Single style mode
        print(f"Single-style mode: {src} -> {tgt}", flush=True)
        count = process_style(src, tgt, max_images)
        print(f"\nFinal: {count} file(s) written for style '{src.name}'", flush=True)
    elif style_subdirs:
        # Multi-style mode
        print(f"Multi-style mode: {src} -> {tgt} ({len(style_subdirs)} style dirs)", flush=True)
        grand_total = 0
        for sd in sorted(style_subdirs, key=lambda p: p.name):
            print(f"\nStyle: {sd.name}", flush=True)
            count = process_style(sd, tgt / sd.name, max_images)
            grand_total += count
        print(f"\nFinal grand total: {grand_total} file(s) written across {len(style_subdirs)} styles", flush=True)
    else:
        print(f"ERROR: no images or style sub-directories found in {src}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
