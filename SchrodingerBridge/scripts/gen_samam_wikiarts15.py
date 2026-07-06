#!/usr/bin/env python3
"""Generate SaMam baseline images for WikiArt-15 (15 styles, distinct5 excluded).

SaMam is image-based: given content image + style image -> stylized output.
Uses pixel-space model (final_model.ckpt) trained on distinct5.

Output structure:
  {output_root}/samam/images/*.png
  {output_root}/samam/_DONE

Naming: {src_style}__{src_stem}__to__{tgt_style}.png
"""
import os
import sys
import time
import random
from pathlib import Path

# Add SaMam repo to sys.path
SAMAM_REPO = r"I:\Github\Latent_Style\Related_Works\repos\SaMam"
sys.path.insert(0, SAMAM_REPO)

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from TEST import test_utils
from TRAIN.lightning_module.lightningmodel import LightningModel

# ── Config ──
TEST_DIR = Path(r"I:\datasets\wikiarts15_512_test")
OUTPUT_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15")
CKPT = os.path.join(SAMAM_REPO, "TRAIN", "final_model.ckpt")

STYLES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque",
    "Color_Field_Painting", "Cubism", "Expressionism", "Fauvism",
    "High_Renaissance", "Mannerism_Late_Renaissance",
    "Naive_Art_Primitivism", "Northern_Renaissance", "Pop_Art",
    "Post_Impressionism", "Romanticism", "Symbolism",
]

IMAGE_SIZE = 512
STYLE_SIZE = 512
MAX_SRC_PER_STYLE = 30
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_model(ckpt_path, device):
    """Load SaMam pixel-space model."""
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=device,
        nVSSMs=2, nSAVSSMs=2, nSAVSSGs=2,
        embed_dim=256, patch_size=8,
        representation_dim=64, d_state=16, expand=2.0,
        compress_ratio=8, squeeze_factor=8,
        mamba_from_trion=1,
    )
    model = model.to(device).eval()
    return model


def stylize(model, content_path, style_path, device, style_size=STYLE_SIZE):
    """Stylize a single content image with a style reference."""
    content_img = test_utils.load(content_path)
    style_img = test_utils.load(style_path)

    content_t = test_utils.content_transforms()(content_img)
    style_t = test_utils.style_transforms(style_size)(style_img)

    content_t = content_t.to(device).unsqueeze(0)
    style_t = style_t.to(device).unsqueeze(0)

    with torch.no_grad():
        output = model.forward(content_t, style_t)
    return output[0].detach().cpu()


def main():
    print(f"=== SaMam WikiArt-15 Inference ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  ckpt: {CKPT}", flush=True)
    print(f"  test_dir: {TEST_DIR}", flush=True)
    print(f"  styles({len(STYLES)}): {STYLES}", flush=True)
    print(f"  image_size: {IMAGE_SIZE}, style_size: {STYLE_SIZE}", flush=True)

    rng = random.Random(SEED)

    # Collect source images per style
    src_images = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"WARNING: {style_dir} not found, skipping", flush=True)
            continue
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(images)
        if MAX_SRC_PER_STYLE > 0:
            images = images[:MAX_SRC_PER_STYLE]
        src_images[style] = images
        print(f"  {style}: {len(images)} source images", flush=True)

    total_src = sum(len(v) for v in src_images.items())
    total = total_src * len(STYLES)
    print(f"  {total_src} srcs x {len(STYLES)} styles = {total} images", flush=True)

    # Collect style reference (first image per style)
    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images:
            style_refs[style] = images[0]
            print(f"  style ref for {style}: {images[0].name}", flush=True)

    out_dir = OUTPUT_ROOT / "samam" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = len(list(out_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    model = load_model(CKPT, device)

    pbar = tqdm(total=total, desc="samam_wikiarts15")
    pbar.update(min(existing, total))

    for src_style, files in src_images.items():
        for src_path in files:
            src_stem = src_path.stem

            for tgt_style in STYLES:
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    pbar.update(1)
                    continue

                style_ref_path = style_refs.get(tgt_style)
                if style_ref_path is None:
                    pbar.update(1)
                    continue

                try:
                    output = stylize(model, src_path, style_ref_path, device, STYLE_SIZE)
                    save_image(output.clamp(0, 1), str(out_path))
                except Exception as e:
                    print(f"\n  ERROR: {src_style}->{tgt_style}: {e}", flush=True)
                pbar.update(1)

    pbar.close()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Write _DONE marker
    done_path = OUTPUT_ROOT / "samam" / "_DONE"
    done_path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    print(f"  _DONE marker written to {done_path}", flush=True)
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
