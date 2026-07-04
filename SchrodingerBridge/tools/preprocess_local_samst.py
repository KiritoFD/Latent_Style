#!/usr/bin/env python3
"""Preprocess local wikiart_samst_5style_train3600 images to SDXL VAE latents.

Usage:
  python tools/preprocess_local_samst.py \
    --raw_root "f:/wikiart_samst_5style_train3600" \
    --output_root "f:/wikiart_samst_5style_512_latents" \
    --target_size 512 \
    --num_test_per_style 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def load_sdxl_vae(device: torch.device):
    """Load SDXL VAE from HuggingFace (or local cache)."""
    from diffusers import AutoencoderKL

    print("Loading SDXL VAE from stabilityai/sdxl-vae...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    print(f"  scaling_factor = {vae.config.scaling_factor}")
    return vae


class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple dataset that returns (image_tensor, stem, style) for all images in a folder."""

    def __init__(self, style_dir: Path, style_name: str, target_size: int):
        self.style_name = style_name
        self.image_files = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            for p in style_dir.glob(ext)
        ) + sorted(
            p for ext in ("*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.BMP")
            for p in style_dir.glob(ext)
        )
        self.image_files = sorted(set(self.image_files))
        self.transform = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, path.stem, self.style_name


@torch.no_grad()
def encode_batch(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images to latents and scale."""
    latent_dist = vae.encode(images).latent_dist
    latent = latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


def process_style(
    vae,
    device: torch.device,
    style_dir: Path,
    style_name: str,
    output_root: Path,
    classview_root: Path,
    target_size: int,
    num_test_per_style: int,
    batch_size: int,
):
    print(f"\nProcessing style: {style_name}")
    ds = ImageFolderDataset(style_dir, style_name, target_size)
    print(f"  Found {len(ds)} images")

    if len(ds) == 0:
        return

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    latent_dir = output_root / style_name
    latent_dir.mkdir(parents=True, exist_ok=True)

    # ClassView test split: first N images as test
    test_dir = classview_root / style_name
    test_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch_tensors, stems, _ in tqdm(loader, desc=f"  encode {style_name}"):
        batch_tensors = batch_tensors.to(device)
        latents = encode_batch(vae, batch_tensors).cpu()

        for i, stem in enumerate(stems):
            latent_path = latent_dir / f"{stem}.pt"
            torch.save(latents[i], latent_path)

            # Copy original image to classview test split if within first N
            if count + i < num_test_per_style:
                src_img = style_dir / f"{stem}.jpg"
                if not src_img.exists():
                    src_img = style_dir / f"{stem}.png"
                if src_img.exists():
                    dst_name = f"{style_name}__{stem}.jpg"
                    shutil.copy2(src_img, test_dir / dst_name)

        count += len(stems)

    print(f"  Saved {count} latents to {latent_dir}")
    print(f"  Saved {min(num_test_per_style, count)} test images to {test_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="f:/wikiart_samst_5style_train3600", type=str)
    parser.add_argument("--output_root", default="f:/wikiart_samst_5style_512_latents", type=str)
    parser.add_argument("--classview_root", default="f:/wikiart_samst_5style_512_classview/test", type=str)
    parser.add_argument("--target_size", default=512,