#!/usr/bin/env python3
"""
Run SDEdit (SD1.5) and SD-Turbo baseline inference on distinct5_512 dataset.

Output structure:
  exp/baseline_v2/images/
    sdedit_str010/   (strength=0.10)
    sdedit_str020/   (strength=0.20)
    sdedit_str035/   (strength=0.35)
    sdedit_str040/   (strength=0.40)
    sdturbo/         (SD-Turbo, 1 step)

Naming: {src_style}__{src_stem}__to__{tgt_style}.png
"""

import gc
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
TEST_DIR = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
OUTPUT_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images")

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

STYLE_PROMPTS = {
    "Early_Renaissance": "a painting in Early Renaissance style",
    "Impressionism": "a painting in Impressionist style",
    "Minimalism": "a painting in Minimalist abstract style",
    "Rococo": "a painting in Rococo ornamental style",
    "Ukiyo_e": "a painting in Ukiyo-e Japanese woodblock print style",
}

NEGATIVE_PROMPT = "blurry, low quality, deformed"

SDEDIT_STRENGTHS = [0.10, 0.20, 0.35, 0.40]
SDEDIT_STEPS = 20
SDEDIT_GUIDANCE = 7.5
SDEDIT_MODEL = "runwayml/stable-diffusion-v1-5"

SDTURBO_STRENGTH = 0.8
SDTURBO_STEPS = 1
SDTURBO_GUIDANCE = 1.0
SDTURBO_MODEL = "stabilityai/sd-turbo"

SEED = 42


def collect_test_images():
    """Collect all test images, grouped by source style."""
    src_images = {}  # {src_style: [list of (stem, full_path)]}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"WARNING: {style_dir} not found, skipping")
            continue
        images = sorted(style_dir.glob("*.jpg")) + sorted(style_dir.glob("*.png"))
        src_images[style] = [(p.stem, p) for p in images]
        print(f"  {style}: {len(images)} images")
    return src_images


def load_image(path, size=512):
    """Load and resize image to size×size."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def build_pipe(model_id, torch_dtype=torch.float16):
    """Load StableDiffusionImg2ImgPipeline with VRAM optimizations."""
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def delete_pipe(pipe):
    """Delete pipeline and free GPU memory."""
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("  Pipeline deleted, GPU memory freed.")


def run_sdedit(src_images):
    """Run SDEdit with SD1.5 at multiple strengths."""
    print("\n" + "=" * 60)
    print("SDEdit (SD 1.5) — Loading pipeline...")
    print("=" * 60)

    pipe = build_pipe(SDEDIT_MODEL)

    for strength in SDEDIT_STRENGTHS:
        subdir = f"sdedit_str{int(strength * 1000):03d}"
        out_dir = OUTPUT_ROOT / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- SDEdit strength={strength:.2f} → {subdir}/ ---")

        # Count existing and total
        total = sum(len(v) for v in src_images.values()) * len(STYLES)
        existing = len(list(out_dir.glob("*.png")))
        if existing >= total:
            print(f"  Already have {existing}/{total} images, skipping.")
            continue

        pbar = tqdm(total=total, desc=f"sdedit_s{int(strength*100)}")
        pbar.update(existing)

        for src_style, files in src_images.items():
            for stem, path in files:
                src_img = load_image(path)

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
                        negative_prompt=NEGATIVE_PROMPT,
                        image=src_img,
                        strength=strength,
                        num_inference_steps=SDEDIT_STEPS,
                        guidance_scale=SDEDIT_GUIDANCE,
                        generator=generator,
                    )

                    result.images[0].save(str(out_path))
                    pbar.update(1)

                # Free source image memory
                del src_img

        pbar.close()

    delete_pipe(pipe)


def run_sdturbo(src_images):
    """Run SD-Turbo (1-step) inference."""
    print("\n" + "=" * 60)
    print("SD-Turbo — Loading pipeline...")
    print("=" * 60)

    pipe = build_pipe(SDTURBO_MODEL)

    out_dir = OUTPUT_ROOT / "sdturbo"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in src_images.values()) * len(STYLES)
    existing = len(list(out_dir.glob("*.png")))
    if existing >= total:
        print(f"  Already have {existing}/{total} images, skipping.")
        delete_pipe(pipe)
        return

    print(f"\n--- SD-Turbo (1 step, strength={SDTURBO_STRENGTH}) → sdturbo/ ---")

    pbar = tqdm(total=total, desc="sdturbo")
    pbar.update(existing)

    for src_style, files in src_images.items():
        for stem, path in files:
            src_img = load_image(path)

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
    delete_pipe(pipe)


def main():
    print("Collecting test images...")
    src_images = collect_test_images()

    total_src = sum(len(v) for v in src_images.values())
    print(f"\nTotal source images: {total_src}")
    print(f"Total outputs per variant: {total_src * len(STYLES)}")
    print(f"SDEdit variants: {len(SDEDIT_STRENGTHS)}")
    print(f"Total images to generate: {total_src * len(STYLES) * (len(SDEDIT_STRENGTHS) + 1)}")
    print(f"Output root: {OUTPUT_ROOT}")

    # Run SDEdit first
    run_sdedit(src_images)

    # Then SD-Turbo
    run_sdturbo(src_images)

    # Final summary
    print("\n" + "=" * 60)
    print("ALL DONE — Summary")
    print("=" * 60)
    for subdir_name in [f"sdedit_str{int(s*1000):03d}" for s in SDEDIT_STRENGTHS] + ["sdturbo"]:
        d = OUTPUT_ROOT / subdir_name
        count = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {subdir_name}/: {count} images")
    print()


if __name__ == "__main__":
    main()
