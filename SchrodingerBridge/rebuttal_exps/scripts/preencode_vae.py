"""Pre-encode training latents with a specified VAE.

Workflow:
  1. Load existing SD1.5 latent cache (.pt files)
  2. Decode each latent back to image using SD1.5 VAE
  3. Re-encode each image with the target VAE (SDXL, TAESD, etc.)
  4. Save as new packed latent cache in the same format

Output format (matches existing):
  {output_dir}/packed/
    00_Early_Renaissance.pt  (dict: schema, subdir, count, files, latents[N,4,64,64])
    01_Impressionism.pt
    ...

Usage:
  python preencode_vae.py --vae sdxl --output_dir data/train_sdxl/.latent_cache
  python preencode_vae.py --vae taesd --output_dir data/train_taesd/.latent_cache
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WEAVE_ROOT = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen")
import os
os.chdir(WEAVE_ROOT)
sys.path.insert(0, str(WEAVE_ROOT))

from utils.inference import load_vae, encode_image, decode_latent

# Existing SD1.5 latent cache
SOURCE_CACHE = WEAVE_ROOT / "data" / "train" / ".latent_cache" / "packed" / "packed"
STYLE_FILES = [
    "00_Early_Renaissance.pt",
    "01_Impressionism.pt",
    "02_Minimalism.pt",
    "03_Rococo.pt",
    "04_Ukiyo_e.pt",
]

# Production SD1.5 scaling
SD15_SCALE = 0.18215


def load_target_vae(vae_name, device, cache_dir):
    """Load target VAE by name."""
    if vae_name == "sdxl":
        vae = load_vae(device=device, model_id="stabilityai/sdxl-vae", cache_dir=cache_dir)
        scale = float(vae.config.scaling_factor)
        latent_ch = 4
    elif vae_name == "taesd":
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                               cache_dir=cache_dir,
                                               torch_dtype=torch.float32).to(device).eval()
        scale = 1.0
        latent_ch = 4
    elif vae_name == "sd15":
        vae = load_vae(device=device, model_id="ema", cache_dir=cache_dir)
        scale = float(vae.config.scaling_factor)
        latent_ch = 4
    else:
        raise ValueError(f"Unknown VAE: {vae_name}")
    print(f"  VAE={vae_name}: scale={scale}, latent_channels={latent_ch}")
    return vae, scale, latent_ch


def decode_latent_to_image(sd15_vae, z, device, batch_size=16):
    """Decode SD1.5 latent to image tensor [-1, 1]."""
    imgs = []
    for i in range(0, z.shape[0], batch_size):
        batch = z[i:i+batch_size]
        # Undo SD1.5 scaling
        batch_unscaled = batch / SD15_SCALE
        with torch.no_grad():
            img = decode_latent(sd15_vae, batch_unscaled, device=device)
        imgs.append(img.cpu())
    return torch.cat(imgs, dim=0)


def encode_image_to_latent(target_vae, img_tensor, device, target_scale, vae_name, batch_size=8):
    """Encode image tensor [-1, 1] to latent with target VAE scaling applied."""
    latents = []
    for i in range(0, img_tensor.shape[0], batch_size):
        batch = img_tensor[i:i+batch_size].to(device)
        with torch.no_grad():
            if vae_name == "taesd":
                # TAESD expects [0, 1] input
                z = target_vae.encode(batch * 0.5 + 0.5).latents
            else:
                z = encode_image(target_vae, batch, device)
            # Apply target VAE scaling
            z = z * target_scale
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True, choices=["sdxl", "taesd", "sd15"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_decode", type=int, default=16)
    parser.add_argument("--batch_encode", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = str(WEAVE_ROOT / "eval_cache" / "hf")
    output_dir = Path(args.output_dir)
    packed_dir = output_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Pre-encode training latents with VAE={args.vae}")
    print(f"Output: {packed_dir}")
    print("=" * 60)

    # Load SD1.5 VAE for decoding existing latents
    print("\nLoading SD1.5 VAE (for decoding existing latents)...")
    sd15_vae = load_vae(device=device, model_id="ema", cache_dir=cache_dir)
    print("  SD1.5 VAE loaded.")

    # Load target VAE for re-encoding
    print(f"\nLoading target VAE: {args.vae}...")
    target_vae, target_scale, latent_ch = load_target_vae(args.vae, device, cache_dir)

    total_count = 0
    for style_file in STYLE_FILES:
        src_path = SOURCE_CACHE / style_file
        if not src_path.exists():
            print(f"\nSKIP: {src_path} not found")
            continue

        print(f"\nProcessing {style_file}...")
        t0 = time.time()
        data = torch.load(src_path, map_location="cpu", weights_only=False)
        style_name = data["subdir"]
        file_list = data["files"]
        latents_sd15 = data["latents"]  # [N, 4, 64, 64], already scaled by SD15_SCALE

        n = latents_sd15.shape[0]
        print(f"  {style_name}: {n} samples, latent shape={latents_sd15.shape}")

        # Step 1: Decode SD1.5 latents to images
        print(f"  Decoding {n} latents to images...")
        t1 = time.time()
        images = decode_latent_to_image(sd15_vae, latents_sd15, device, batch_size=args.batch_decode)
        print(f"  Decoded in {time.time()-t1:.1f}s, image shape={images.shape}")

        # Step 2: Re-encode images with target VAE
        print(f"  Re-encoding {n} images with {args.vae} VAE...")
        t2 = time.time()
        new_latents = encode_image_to_latent(
            target_vae, images, device, target_scale, args.vae, batch_size=args.batch_encode)
        print(f"  Encoded in {time.time()-t2:.1f}s, new latent shape={new_latents.shape}")

        # Step 3: Save in same format
        out_data = {
            "schema": data["schema"],
            "subdir": style_name,
            "count": n,
            "files": file_list,
            "latents": new_latents,
        }
        out_path = packed_dir / style_file
        torch.save(out_data, out_path)
        print(f"  Saved to {out_path} ({time.time()-t0:.1f}s total)")
        total_count += n

    # Save metadata
    meta = {
        "vae": args.vae,
        "vae_scale": target_scale,
        "latent_channels": latent_ch,
        "total_samples": total_count,
        "source_cache": str(SOURCE_CACHE),
        "sd15_scale": SD15_SCALE,
    }
    (output_dir / "meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"DONE: {total_count} samples encoded with {args.vae} VAE")
    print(f"Output: {packed_dir}")
    print(f"VAE scale: {target_scale}")
    print(f"{'='*60}")
    print("PREENCODE_EXIT=0")


if __name__ == "__main__":
    main()
