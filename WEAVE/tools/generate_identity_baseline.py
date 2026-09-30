#!/usr/bin/env python3
"""
Generate identity mapping baseline: copy source images as-is (no style transfer).
For each style pair (src_style -> tgt_style), including identity pairs,
take up to 30 source images from the src_style test directory and copy them
with the standard naming format: {src_style}__{src_style}__{src_name}__to__{tgt_style}.png

This creates 5x5x30 = 750 images where the "transfer" is just the identity (no change).
"""
import os
import shutil
from pathlib import Path
from PIL import Image

STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
OUTPUT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\identity")
MAX_IMAGES = 30


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for src_style in STYLE_NAMES:
        src_dir = TEST_DIR / src_style
        if not src_dir.is_dir():
            print(f"  SKIP: {src_dir} not found")
            continue

        # Collect image files (jpg/png), take first MAX_IMAGES
        images = sorted(
            [f for f in src_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )[:MAX_IMAGES]

        if len(images) < MAX_IMAGES:
            print(f"  WARNING: {src_style} only has {len(images)} images (expected {MAX_IMAGES})")

        for img_path in images:
            # Parse stem: Style__artist_title
            stem = img_path.stem
            if "__" in stem:
                prefix, src_name = stem.split("__", 1)
            else:
                prefix = src_style
                src_name = stem

            for tgt_style in STYLE_NAMES:
                out_name = f"{src_style}__{src_style}__{src_name}__to__{tgt_style}.png"
                out_path = OUTPUT_DIR / out_name
                if not out_path.exists():
                    # Convert to PNG
                    img = Image.open(img_path).convert("RGB")
                    img.save(out_path, "PNG")
                total += 1

        print(f"  {src_style}: {len(images)} images x 5 targets = {len(images)*5} outputs")

    print(f"\nTotal images generated: {total}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
