#!/usr/bin/env python
r"""Pre-encode 256x256 images as SDXL VAE latents (.pt tensors) for the latent-256 SFM ablation.

Walks a source directory of images (*.jpg / *.png), resizes each to 256x256 with LANCZOS,
converts to a float32 tensor of shape (3, 256, 256) normalized to [-1, 1], encodes through
the SDXL VAE (AutoencoderKL from "stabilityai/sdxl-vae"), and saves the resulting 4x16x16
latent (scaled by latent_scale_factor=0.18215) as a .pt file so the existing
AdaCUTLatentDataset loader (which only accepts .pt/.npy) can consume them without any
code changes.

The SDXL VAE downsamples by 8x. CPU-only by default (use --device cuda if available).
Supports resume (skips existing .pt).

Two usage modes:
  1) Single style directory:
     python scripts/pre_encode_latent256.py \
         --source "F:\wikiart_distinct5_samam_512_classview_real\train\Early_Renaissance" \
         --target "F:\wikiart_distinct5_samam_512_latent256\train\Early_Renaissance" \
         --max 50

  2) Parent directory with multiple style sub-directories:
     python scripts/pre_encode_latent256.py \
         --source "F:\wikiart_distinct5_samam_512_classview_real\train" \
         --target "F:\wikiart_distinct5_samam_512_latent256\train"
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
LATENT_SCALE_FACTOR = 0.18215  # standard SDXL scaling


def _load_vae(vae_path: str, cache_dir: str, device: str):
    """Load SDXL VAE (AutoencoderKL) from *vae_path* (HF repo id or local path)."""
    try:
        # Prefer the legacy direct module path first. Some newer diffusers
        # package-level imports eagerly import optional autoencoder families
        # that require newer transformers than the local env may have.
        from diffusers.models.autoencoder_kl import AutoencoderKL
    except Exception:
        from diffusers import AutoencoderKL

    kwargs = {"torch_dtype": torch.float32}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kwargs["cache_dir"] = cache_dir

    # If vae_path is a local directory, load directly. Otherwise try HF cache
    # (local_files_only) first, then fall back to a real download.
    if vae_path and os.path.isdir(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, **kwargs)
    else:
        try:
            vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, **kwargs)
        except Exception:
            vae = AutoencoderKL.from_pretrained(vae_path, **kwargs)

    vae.eval()
    vae.to(device)
    return vae


def encode_one(img_path: Path, vae, device: str) -> torch.Tensor:
    """Load image, resize to 256x256 LANCZOS, encode through VAE, return float32 latent (4, 16, 16)."""
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()  # (3, H, W)
    tensor = tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample()
    latent = latent * LATENT_SCALE_FACTOR
    latent = latent.squeeze(0).cpu().float().contiguous()  # (4, H/8, W/8)
    return latent


def process_style(src_dir: Path, tgt_dir: Path, max_images: int, vae, device: str) -> int:
    """Encode all images in *src_dir* into *tgt_dir* as .pt latents. Returns count written."""
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
            tensor = encode_one(img_path, vae, device)
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
        description="Pre-encode 256x256 images as SDXL VAE latents (.pt) for latent-256 SFM ablation"
    )
    parser.add_argument("--source", type=str, required=True, help="Source image dir (single style or parent of styles)")
    parser.add_argument("--target", type=str, required=True, help="Target .pt output dir (single style or parent of styles)")
    parser.add_argument("--max", type=int, default=0, help="Max images per style (0 = all). For smoke testing.")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sdxl-vae",
                        help="VAE model id or local path (default: stabilityai/sdxl-vae). "
                             "Point to a local cache dir to avoid HF download.")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="HuggingFace cache dir for VAE download (default: HF default cache).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for VAE encode (default: cpu; use 'cuda' if available).")
    args = parser.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    max_images = max(0, int(args.max))
    vae_path = args.vae_path
    cache_dir = args.cache_dir
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.", file=sys.stderr, flush=True)
        device = "cpu"

    if not src.is_dir():
        print(f"ERROR: source dir does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loading VAE from {vae_path} (cache_dir={cache_dir or 'HF default'}, device={device})...",
        flush=True,
    )
    vae = _load_vae(vae_path, cache_dir, device)
    print("VAE loaded.", flush=True)

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
        count = process_style(src, tgt, max_images, vae, device)
        print(f"\nFinal: {count} file(s) written for style '{src.name}'", flush=True)
    elif style_subdirs:
        # Multi-style mode
        print(f"Multi-style mode: {src} -> {tgt} ({len(style_subdirs)} style dirs)", flush=True)
        grand_total = 0
        for sd in sorted(style_subdirs, key=lambda p: p.name):
            print(f"\nStyle: {sd.name}", flush=True)
            count = process_style(sd, tgt / sd.name, max_images, vae, device)
            grand_total += count
        print(f"\nFinal grand total: {grand_total} file(s) written across {len(style_subdirs)} styles", flush=True)
    else:
        print(f"ERROR: no images or style sub-directories found in {src}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
