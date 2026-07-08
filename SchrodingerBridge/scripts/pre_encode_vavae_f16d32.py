#!/usr/bin/env python
r"""Pre-encode images as VAVAE f16d32 latents (.pt tensors).

VAVAE (hustvl/vavae-imagenet256-f16d32-dinov2):
  - 32-channel latents, 16x downsampling
  - Trained on ImageNet-256 (native resolution 256x256)
  - Uses LDM-style AutoencoderKL (not diffusers')
  - No scaling factor (raw latent output, unlike SD's 0.18215)

Input: images resized to 256x256 (VAVAE's native resolution).
Output: per-image .pt tensor of shape (32, 16, 16), float32.

Usage:
  python scripts/pre_encode_vavae_f16d32.py \
      --source "I:/wikiart_distinct5_samam_512_classview/train" \
      --target "I:/wikiart_distinct5_samam_512_vavae_f16d32/train" \
      --max 50

  # Full encoding (no max limit):
  python scripts/pre_encode_vavae_f16d32.py \
      --source "I:/wikiart_distinct5_samam_512_classview/train" \
      --target "I:/wikiart_distinct5_samam_512_vavae_f16d32/train"
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
# VAVAE native resolution is 256, but it can handle any resolution.
# Use 256 for quick validation; switch to 512 for full experiment.
SIZE = 256
# VAVAE outputs raw latents (no scaling factor like SD's 0.18215)
LATENT_SCALE_FACTOR = 1.0
# VAVAE has 32 latent channels
LATENT_CHANNELS = 32
# Downsampling factor
DOWNSAMPLE_FACTOR = 16


def _load_vavae(vae_path: str | Path, device: str):
    """Load VAVAE model from LightningDiT/LDM checkpoint."""
    vae_path = Path(vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

    # Add local _vavae_repo/tokenizer to path
    _vavae_root = Path(__file__).resolve().parent.parent / "_vavae_repo" / "tokenizer"
    if str(_vavae_root) not in sys.path:
        sys.path.insert(0, str(_vavae_root))

    from autoencoder import AutoencoderKL as VAVAE_AutoencoderKL

    model = VAVAE_AutoencoderKL(
        embed_dim=32,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=str(vae_path),
        model_type="vavae",
    )
    model.eval().to(device)
    return model, "vavae_autoencoder"


def _load_vavae_via_huggingface(device: str, cache_dir: str = ""):
    """Load VAVAE using huggingface hub download + manual state dict loading.

    This downloads the .pt file from HF and constructs the model.
    """
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id="hustvl/vavae-imagenet256-f16d32-dinov2",
        filename="vavae-imagenet256-f16d32-dinov2.pt",
        cache_dir=cache_dir or None,
    )
    return _load_vavae(ckpt_path, device)


def encode_one(img_path: Path, vae, device: str) -> torch.Tensor:
    """Load image, resize to 256x256, encode through VAVAE, return float32 latent."""
    img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        posterior = vae.encode(tensor)
        latent = posterior.sample()
    latent = latent * LATENT_SCALE_FACTOR
    latent = latent.squeeze(0).cpu().float().contiguous()  # (C, H//16, W//16)
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
        description="Pre-encode images as VAVAE f16d32 latents (.pt)"
    )
    parser.add_argument("--source", type=str, required=True, help="Source image dir")
    parser.add_argument("--target", type=str, required=True, help="Target .pt output dir")
    parser.add_argument("--max", type=int, default=0, help="Max images per style (0=all)")
    parser.add_argument("--ckpt", type=str, default="", help="Local path to VAVAE .pt checkpoint (skip HF download)")
    parser.add_argument("--cache_dir", type=str, default="", help="HF cache dir for download")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for VAE (default: cuda)")
    args = parser.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    max_images = max(0, int(args.max))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.", file=sys.stderr, flush=True)
        device = "cpu"

    if not src.is_dir():
        print(f"ERROR: source dir does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    # Load VAVAE
    print("Loading VAVAE...", flush=True)
    if args.ckpt and Path(args.ckpt).exists():
        vae, loader_type = _load_vavae(args.ckpt, device)
        print(f"  VAVAE loaded from local checkpoint ({loader_type})", flush=True)
    else:
        vae, loader_type = _load_vavae_via_huggingface(device, args.cache_dir or None)
        print(f"  VAVAE downloaded and loaded ({loader_type})", flush=True)

    # Verify output shape
    test_input = torch.randn(1, 3, SIZE, SIZE, device=device)
    with torch.no_grad():
        test_out = vae.encode(test_input).sample()
    print(f"  VAVAE output shape: {test_out.shape} (expected [B, {LATENT_CHANNELS}, {SIZE//DOWNSAMPLE_FACTOR}, {SIZE//DOWNSAMPLE_FACTOR}])", flush=True)

    # Detect mode
    entries = list(os.scandir(src))
    direct_images = [Path(e.path) for e in entries
                     if Path(e.name).suffix.lower() in IMAGE_EXTS and e.is_file()]
    style_subdirs = [Path(e.path) for e in entries
                     if e.is_dir() and not e.name.startswith(".") and not e.name.startswith(".latent")]

    if direct_images:
        print(f"Single-style mode: {src} -> {tgt}", flush=True)
        count = process_style(src, tgt, max_images, vae, device)
        print(f"\nFinal: {count} file(s) written for style '{src.name}'", flush=True)
    elif style_subdirs:
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
