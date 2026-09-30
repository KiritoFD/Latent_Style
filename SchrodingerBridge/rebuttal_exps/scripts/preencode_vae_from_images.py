"""Pre-encode training latents DIRECTLY from raw images with a target VAE.

This avoids the decode-then-reencode error accumulation that the previous
preencode_vae.py suffered from. Raw images live at:
    F:\\wikiart_distinct5_samam_512_classview_real\\train\\{style}\\{name}.jpg

Output format (matches existing packed cache consumed by AdaCUTLatentDataset):
    {output_dir}/packed/
        00_Early_Renaissance.pt  (dict: schema, subdir, count, files, latents[N,4,64,64])
        01_Impressionism.pt
        ...

The output latents are ALREADY scaled by the target VAE's scaling_factor,
consistent with how the SD1.5 packed cache was produced.

Usage:
    python preencode_vae_from_images.py --vae sdxl  --output_dir data/train_sdxl/.latent_cache
    python preencode_vae_from_images.py --vae taesd --output_dir data/train_taesd/.latent_cache
    python preencode_vae_from_images.py --vae sd15  --output_dir data/train_sd15_reencode/.latent_cache
"""
import argparse
import json
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

from utils.inference import load_vae, encode_image

# Raw image dataset (verified 100% aligned with existing packed latent cache)
IMG_ROOT = Path(r"F:\wikiart_distinct5_samam_512_classview_real\train")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_FILES = [
    "00_Early_Renaissance.pt",
    "01_Impressionism.pt",
    "02_Minimalism.pt",
    "03_Rococo.pt",
    "04_Ukiyo_e.pt",
]

# Reference packed cache (only used to read the canonical file list / order,
# so the new cache has exactly the same sample ordering as the SD1.5 one).
REF_PACKED_ROOT = WEAVE_ROOT / "data" / "train" / ".latent_cache" / "packed" / "packed"

IMAGE_SIZE = 512  # raw images are 512x512


def load_target_vae(vae_name: str, device, cache_dir: str):
    """Load target VAE by name. Returns (vae, scale, latent_ch, needs_unit_input)."""
    if vae_name == "sdxl":
        vae = load_vae(device=device, model_id="stabilityai/sdxl-vae", cache_dir=cache_dir)
        scale = float(vae.config.scaling_factor)
        latent_ch = 4
        needs_unit = False
    elif vae_name == "sd15":
        vae = load_vae(device=device, model_id="ema", cache_dir=cache_dir)
        scale = float(vae.config.scaling_factor)
        latent_ch = 4
        needs_unit = False
    elif vae_name == "taesd":
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
        ).to(device).eval()
        scale = 1.0
        latent_ch = 4
        needs_unit = True  # TAESD expects [0, 1] input
    else:
        raise ValueError(f"Unknown VAE: {vae_name}")
    print(f"  VAE={vae_name}: scale={scale}, latent_channels={latent_ch}, needs_unit_input={needs_unit}")
    return vae, scale, latent_ch, needs_unit


def load_image_tensor(path: Path, device, dtype=torch.float32) -> torch.Tensor:
    """Load an image as a [1, C, H, W] tensor in [-1, 1] range (float32 on CPU then moved)."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W] in [0, 1]
    t = (t * 2.0) - 1.0  # to [-1, 1]
    return t.unsqueeze(0).to(device)


def encode_images_with_vae(
    vae,
    image_paths: list[Path],
    device,
    vae_name: str,
    target_scale: float,
    needs_unit: bool,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode a list of images into latents [N, C, H, W] (already scaled, float32, on CPU)."""
    out_latents = []
    for i in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[i : i + batch_size]
        tensors = []
        for p in chunk_paths:
            tensors.append(load_image_tensor(p, device))
        batch = torch.cat(tensors, dim=0)  # [B, 3, H, W] in [-1, 1]
        with torch.no_grad():
            if vae_name == "taesd":
                # TAESD expects [0, 1] input; encode returns .latents (no scaling factor in config)
                z = vae.encode(batch * 0.5 + 0.5).latents
                z = z * target_scale  # target_scale is 1.0 for TAESD, kept for symmetry
            else:
                # encode_image already multiplies by vae.config.scaling_factor
                z = encode_image(vae, batch, device=device)
        out_latents.append(z.detach().cpu().float())
        if (i // batch_size) % 10 == 0:
            print(f"    encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)
    return torch.cat(out_latents, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True, choices=["sdxl", "taesd", "sd15"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_encode", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = str(WEAVE_ROOT / "eval_cache" / "hf")
    output_dir = Path(args.output_dir)
    packed_dir = output_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Pre-encode training latents DIRECTLY from raw images")
    print(f"  VAE:       {args.vae}")
    print(f"  Image root: {IMG_ROOT}")
    print(f"  Output:     {packed_dir}")
    print("=" * 70)

    # Load target VAE
    print(f"\nLoading target VAE: {args.vae}...")
    target_vae, target_scale, latent_ch, needs_unit = load_target_vae(
        args.vae, device, cache_dir
    )

    total_count = 0
    for style, packed_file in zip(STYLES, STYLE_FILES):
        # Load reference packed cache to get the canonical file ordering
        ref_path = REF_PACKED_ROOT / packed_file
        if not ref_path.exists():
            print(f"\nSKIP: reference packed cache {ref_path} not found")
            continue
        ref_payload = torch.load(ref_path, map_location="cpu", weights_only=False)
        ref_files = ref_payload["files"]  # list of relative paths like "Early_Renaissance/Early_Renaissance__xxx.pt"
        # Convert each ref file entry to the actual image path
        image_paths = []
        missing = []
        for rel in ref_files:
            # rel is like "Early_Renaissance/Early_Renaissance__xxx.pt" (with .pt extension)
            # Strip directory prefix and .pt extension to get the stem, then look for image
            parts = rel.replace("\\", "/").split("/")
            stem = parts[-1]
            if stem.endswith(".pt"):
                stem = stem[:-3]
            elif stem.endswith(".npy"):
                stem = stem[:-4]
            found = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = IMG_ROOT / style / (stem + ext)
                if cand.exists():
                    found = cand
                    break
            if found is None:
                missing.append(rel)
            else:
                image_paths.append(found)
        if missing:
            print(f"\nWARN: {len(missing)} images missing for {style}; first: {missing[:3]}")

        n = len(image_paths)
        print(f"\nProcessing {packed_file}  (style={style}, n={n})")
        t0 = time.time()

        # Encode all images directly with target VAE
        print(f"  Encoding {n} images with {args.vae} VAE (batch={args.batch_encode})...")
        t1 = time.time()
        new_latents = encode_images_with_vae(
            target_vae,
            image_paths,
            device,
            vae_name=args.vae,
            target_scale=target_scale,
            needs_unit=needs_unit,
            batch_size=args.batch_encode,
        )
        print(
            f"  Encoded in {time.time()-t1:.1f}s, latent shape={new_latents.shape}, "
            f"range=[{new_latents.min():.3f}, {new_latents.max():.3f}]"
        )

        # Save in the same packed format
        out_data = {
            "schema": ref_payload["schema"],
            "subdir": style,
            "count": n,
            "files": ref_files,  # keep canonical file list
            "latents": new_latents,
        }
        out_path = packed_dir / packed_file
        torch.save(out_data, out_path)
        print(f"  Saved to {out_path}  ({time.time()-t0:.1f}s total)")
        total_count += n

        # Free memory
        del new_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save metadata
    meta = {
        "vae": args.vae,
        "vae_scale": target_scale,
        "latent_channels": latent_ch,
        "total_samples": total_count,
        "image_root": str(IMG_ROOT),
        "image_size": IMAGE_SIZE,
        "encode_method": "direct_from_images",
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"DONE: {total_count} samples encoded with {args.vae} VAE (direct from images)")
    print(f"Output: {packed_dir}")
    print(f"VAE scale: {target_scale}")
    print(f"{'='*70}")
    print("PREENCODE_FROM_IMAGES_EXIT=0")


if __name__ == "__main__":
    main()
