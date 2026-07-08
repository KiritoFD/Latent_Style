#!/usr/bin/env python
r"""Pre-encode images as latents (.pt tensors) for multiple VAE backends.

Supported VAEs:
  - sd15:  stabilityai/sd-vae-ft-mse (4ch, f8, scale=0.18215)
  - sdxl:  stabilityai/sdxl-vae      (4ch, f8, scale=0.13025)
  - flux:  black-forest-labs/FLUX.1-schnell AE (16ch, f8, scale=0.3611)
  - vavae: hustvl/vavae-imagenet256-f16d32-dinov2 (32ch, f16, scale=1.0)

Usage:
  python scripts/pre_encode_multi_vae.py \
      --vae sdxl \
      --source "F:/wikiart_distinct5_512_images/train" \
      --target "G:/GitHub/Latent_Style/Dataset/distinct5_512_sdxl/train" \
      --size 512

  python scripts/pre_encode_multi_vae.py \
      --vae flux \
      --source "F:/wikiart_distinct5_512_images/train" \
      --target "G:/GitHub/Latent_Style/Dataset/distinct5_512_flux/train" \
      --size 512

  python scripts/pre_encode_multi_vae.py \
      --vae vavae \
      --source "F:/wikiart_distinct5_512_images/train" \
      --target "G:/GitHub/Latent_Style/Dataset/distinct5_512_vavae/train" \
      --size 512
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

VAE_SPECS = {
    "sd15": {
        "hf_id": "stabilityai/sd-vae-ft-mse",
        "latent_channels": 4,
        "downsample": 8,
        "scaling_factor": 0.18215,
        "dtype": "float16",
        "loader": "diffusers",
    },
    "sdxl": {
        "hf_id": "stabilityai/sdxl-vae",
        "latent_channels": 4,
        "downsample": 8,
        "scaling_factor": 0.13025,
        "dtype": "float16",
        "loader": "diffusers",
    },
    "flux": {
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "subfolder": "ae",
        "latent_channels": 16,
        "downsample": 8,
        "scaling_factor": 0.3611,
        "dtype": "float16",
        "loader": "diffusers",
    },
    "vavae": {
        "hf_id": "hustvl/vavae-imagenet256-f16d32-dinov2",
        "hf_filename": "vavae-imagenet256-f16d32-dinov2.pt",
        "latent_channels": 32,
        "downsample": 16,
        "scaling_factor": 1.0,
        "dtype": "float32",
        "loader": "vavae",
    },
}


def load_vae(vae_name: str, device: str, cache_dir: str = ""):
    """Load a VAE model by name. Returns (model, spec_dict)."""
    spec = VAE_SPECS[vae_name]
    loader = spec["loader"]

    if loader == "vavae":
        from huggingface_hub import hf_hub_download
        _vavae_root = os.path.join(os.path.dirname(__file__), "..", "_vavae_repo", "tokenizer")
        _vavae_root = os.path.abspath(_vavae_root)
        if _vavae_root not in sys.path:
            sys.path.insert(0, _vavae_root)
        from autoencoder import AutoencoderKL as VAVAE_AutoencoderKL

        ckpt_path = hf_hub_download(
            repo_id=spec["hf_id"],
            filename=spec["hf_filename"],
            cache_dir=cache_dir or None,
        )
        model = VAVAE_AutoencoderKL(
            embed_dim=32, ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=ckpt_path, model_type="vavae",
        )
        model.eval().to(device)
        return model, spec

    # diffusers-based VAEs (sd15, sdxl, flux)
    try:
        from diffusers.models.autoencoder_kl import AutoencoderKL
    except Exception:
        from diffusers import AutoencoderKL

    hf_id = spec["hf_id"]
    subfolder = spec.get("subfolder", None)
    # Always load in fp32 for stable encoding (fp16 VAE can produce NaN)
    dtype = torch.float32

    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        if subfolder:
            vae = AutoencoderKL.from_pretrained(hf_id, subfolder=subfolder, **kwargs).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(hf_id, **kwargs).to(device)
    except Exception:
        # Try without local_files_only
        kwargs.pop("local_files_only", None)
        if subfolder:
            vae = AutoencoderKL.from_pretrained(hf_id, subfolder=subfolder, **kwargs).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(hf_id, **kwargs).to(device)

    vae.eval()
    # Disable tiling/slicing for consistent encoding
    for method_name in ("disable_slicing", "disable_tiling"):
        method = getattr(vae, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    return vae, spec


def encode_one(img_path: Path, vae, spec: dict, device: str, size: int) -> torch.Tensor:
    """Load image, resize, encode through VAE, return float32 latent."""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    dtype = getattr(torch, spec["dtype"])
    loader = spec["loader"]
    scale = spec["scaling_factor"]

    with torch.no_grad():
        if loader == "vavae":
            # VAVAE requires float32 input
            tensor = tensor.to(dtype=torch.float32)
            latent = vae.encode(tensor).sample()
        else:
            # diffusers AutoencoderKL — always encode in fp32 for stability
            tensor = tensor.to(dtype=torch.float32)
            if hasattr(vae, 'encode'):
                try:
                    latent = vae.encode(tensor).latent_dist.sample()
                except Exception:
                    # Fallback for some VAE variants
                    out = vae.encode(tensor)
                    latent = getattr(out, 'sample', lambda: out.latents)()
                    if isinstance(latent, tuple):
                        latent = latent[0]
                latent = latent * scale
            else:
                raise RuntimeError(f"Unsupported VAE type: {type(vae)}")

    latent = latent.squeeze(0).cpu().float().contiguous()
    return latent


def process_style(src_dir: Path, tgt_dir: Path, max_images: int, vae, spec: dict, device: str, size: int) -> int:
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
            tensor = encode_one(img_path, vae, spec, device, size)
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
    parser = argparse.ArgumentParser(description="Pre-encode images as latents for multiple VAE backends")
    parser.add_argument("--vae", type=str, required=True, choices=list(VAE_SPECS.keys()),
                        help="VAE type: sd15, sdxl, flux, vavae")
    parser.add_argument("--source", type=str, required=True, help="Source image dir")
    parser.add_argument("--target", type=str, required=True, help="Target .pt output dir")
    parser.add_argument("--size", type=int, default=512, help="Image resize (default: 512)")
    parser.add_argument("--max", type=int, default=0, help="Max images per style (0=all)")
    parser.add_argument("--cache_dir", type=str, default="", help="HF cache dir")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
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

    spec = VAE_SPECS[args.vae]
    size = args.size
    expected_h = size // spec["downsample"]
    expected_w = size // spec["downsample"]
    expected_ch = spec["latent_channels"]

    print(f"Loading {args.vae} VAE (hf_id={spec['hf_id']})...", flush=True)
    vae, spec = load_vae(args.vae, device, args.cache_dir or "")

    # Verify output shape
    test_input = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
    with torch.no_grad():
        if spec.get("loader") == "vavae":
            test_out = vae.encode(test_input).sample()
        else:
            test_out = vae.encode(test_input).latent_dist.sample()
            test_out = test_out * spec["scaling_factor"]
    print(f"  VAE output shape: {test_out.shape} (expected [1, {expected_ch}, {expected_h}, {expected_w}])", flush=True)

    # Detect mode
    entries = list(os.scandir(src))
    direct_images = [Path(e.path) for e in entries
                     if Path(e.name).suffix.lower() in IMAGE_EXTS and e.is_file()]
    style_subdirs = [Path(e.path) for e in entries
                     if e.is_dir() and not e.name.startswith(".") and not e.name.startswith(".latent")]

    if direct_images:
        print(f"Single-style mode: {src} -> {tgt}", flush=True)
        count = process_style(src, tgt, max_images, vae, spec, device, size)
        print(f"\nFinal: {count} file(s) written for style '{src.name}'", flush=True)
    elif style_subdirs:
        print(f"Multi-style mode: {src} -> {tgt} ({len(style_subdirs)} style dirs)", flush=True)
        grand_total = 0
        for sd in sorted(style_subdirs, key=lambda p: p.name):
            print(f"\nStyle: {sd.name}", flush=True)
            count = process_style(sd, tgt / sd.name, max_images, vae, spec, device, size)
            grand_total += count
        print(f"\nFinal grand total: {grand_total} file(s) written across {len(style_subdirs)} styles", flush=True)
    else:
        print(f"ERROR: no images or style sub-directories found in {src}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
