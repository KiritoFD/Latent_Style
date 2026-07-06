#!/usr/bin/env python3
"""Generate SD-Turbo baseline images for WikiArt-15 (15 styles, distinct5 excluded).

Output structure:
  {output_root}/sdturbo/images/*.png
  {output_root}/sdturbo/_DONE

Naming: {src_style}__{src_stem}__to__{tgt_style}.png
"""
import gc
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── Config ──
TEST_DIR = Path(r"I:\datasets\wikiarts15_512_test")
OUTPUT_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15")

STYLES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque",
    "Color_Field_Painting", "Cubism", "Expressionism", "Fauvism",
    "High_Renaissance", "Mannerism_Late_Renaissance",
    "Naive_Art_Primitivism", "Northern_Renaissance", "Pop_Art",
    "Post_Impressionism", "Romanticism", "Symbolism",
]

STYLE_PROMPTS = {
    "Abstract_Expressionism": "a painting in Abstract Expressionism style, bold gestural brushstrokes",
    "Art_Nouveau_Modern": "a painting in Art Nouveau style, ornamental decorative lines",
    "Baroque": "a painting in Baroque style, dramatic chiaroscuro lighting",
    "Color_Field_Painting": "a painting in Color Field style, large flat areas of color",
    "Cubism": "a painting in Cubism style, geometric fragmented forms",
    "Expressionism": "a painting in Expressionism style, vivid emotional colors",
    "Fauvism": "a painting in Fauvism style, wild unnatural vibrant colors",
    "High_Renaissance": "a painting in High Renaissance style, balanced classical composition",
    "Mannerism_Late_Renaissance": "a painting in Mannerism style, elongated figures complex poses",
    "Naive_Art_Primitivism": "a painting in Naive Art Primitivism style, simple childlike forms",
    "Northern_Renaissance": "a painting in Northern Renaissance style, detailed realistic oil technique",
    "Pop_Art": "a painting in Pop Art style, bold commercial imagery",
    "Post_Impressionism": "a painting in Post-Impressionism style, structured brushwork vivid color",
    "Romanticism": "a painting in Romanticism style, dramatic emotional sublime scenery",
    "Symbolism": "a painting in Symbolism style, dreamlike metaphorical imagery",
}

SDTURBO_MODEL = "stabilityai/sd-turbo"
SDTURBO_STRENGTH = 0.8
SDTURBO_STEPS = 1
SDTURBO_GUIDANCE = 1.0
SEED = 42
IMAGE_SIZE = 512
MAX_SRC_PER_STYLE = 30

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_test_images():
    """Collect test images, grouped by source style."""
    import random
    rng = random.Random(SEED)
    src_images = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"WARNING: {style_dir} not found, skipping")
            continue
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(images)
        if MAX_SRC_PER_STYLE > 0:
            images = images[:MAX_SRC_PER_STYLE]
        src_images[style] = images
        print(f"  {style}: {len(images)} images")
    return src_images


def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def build_pipe(model_id, torch_dtype=torch.float16):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def main():
    print(f"=== SD-Turbo WikiArt-15 Inference ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    src_images = collect_test_images()
    total_src = sum(len(v) for v in src_images.values())
    total = total_src * len(STYLES)
    print(f"  {total_src} srcs x {len(STYLES)} styles = {total} images", flush=True)

    out_dir = OUTPUT_ROOT / "sdturbo" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = len(list(out_dir.glob("*.png")))
    if existing >= total:
        print(f"  Already have {existing}/{total} images, skipping.", flush=True)
    else:
        print(f"  Loading SD-Turbo pipeline...", flush=True)
        pipe = build_pipe(SDTURBO_MODEL)

        pbar = tqdm(total=total, desc="sdturbo_wikiarts15")
        pbar.update(existing)

        for src_style, files in src_images.items():
            for path in files:
                src_img = load_image(path)
                stem = path.stem

                for tgt_style in STYLES:
                    out_name = f"{src_style}__{stem}__to__{tgt_style}.png"
                    out_path = out_dir / out_name

                    if out_path.exists():
                        pbar.update(1)
                        continue

                    prompt = STYLE_PROMPTS[tgt_style]
                    generator = torch.Generator("cuda").manual_seed(SEED)

                    result = pipe(
                        prompt=prompt,
                        image=src_img,
                        strength=SDTURBO_STRENGTH,
                        num_inference_steps=SDTURBO_STEPS,
                        guidance_scale=SDTURBO_GUIDANCE,
                        generator=generator,
                    )
                    result.images[0].save(str(out_path))
                    pbar.update(1)

                del src_img

        pbar.close()
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Write _DONE marker
    done_path = OUTPUT_ROOT / "sdturbo" / "_DONE"
    done_path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    print(f"  _DONE marker written to {done_path}", flush=True)
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
