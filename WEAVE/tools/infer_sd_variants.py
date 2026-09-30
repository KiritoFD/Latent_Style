"""Inference script for baseline image generation using SDEdit and SD-Turbo.

Generates style-transferred images on the distinct5_512 test set for
comparison with the Schrödinger Bridge method.

Usage:
    python tools/infer_sd_variants.py --method sdedit --strength 0.10
    python tools/infer_sd_variants.py --method sdturbo
    python tools/infer_sd_variants.py --method all
    python tools/infer_sd_variants.py --method sdedit --strength 0.35 --max_pairs 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

DATASET_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512")
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images")

SDEDIT_MODEL = "runwayml/stable-diffusion-v1-5"
SDTURBO_MODEL = "stabilityai/sd-turbo"

SDEDIT_STRENGTHS = [0.10, 0.20, 0.35, 0.40]

NUM_IMAGES_PER_PAIR = 30
BASE_SEED = 42
IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_pairs():
    """Build all 5x5 source->target pairs (including identity pairs)."""
    pairs = []
    for src in STYLE_NAMES:
        for tgt in STYLE_NAMES:
            pairs.append((src, tgt))
    return pairs


def get_test_images(style_name: str) -> list[Path]:
    """Return sorted list of test image paths for a given style."""
    style_dir = TEST_DIR / style_name
    if not style_dir.exists():
        print(f"[WARN] Test directory not found: {style_dir}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in style_dir.iterdir() if p.suffix.lower() in exts)


def src_name_from_filename(filename: str) -> str:
    """Extract the artist_title part from '{Style}__{artist}_{title}.jpg'.

    Returns the part after the first '__' without the extension.
    """
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


# ---------------------------------------------------------------------------
# SDEdit
# ---------------------------------------------------------------------------
def run_sdedit(strengths: list[float], pairs: list, device: str):
    from diffusers import StableDiffusionImg2ImgPipeline

    print(f"\n{'='*60}")
    print(f"SDEdit  |  model: {SDEDIT_MODEL}")
    print(f"{'='*60}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SDEDIT_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_model_cpu_offload()

    for strength in strengths:
        out_dir = OUTPUT_ROOT / f"sdedit_str{strength:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- SDEdit strength={strength:.2f} -> {out_dir} ---")

        _generate_all_pairs(pipe, pairs, out_dir, strength=strength, device=device)

    # Free memory
    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# SD-Turbo
# ---------------------------------------------------------------------------
def run_sdturbo(pairs: list, device: str):
    from diffusers import StableDiffusionImg2ImgPipeline

    print(f"\n{'='*60}")
    print(f"SD-Turbo  |  model: {SDTURBO_MODEL}")
    print(f"{'='*60}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SDTURBO_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_model_cpu_offload()

    out_dir = OUTPUT_ROOT / "sdturbo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SD-Turbo: use num_inference_steps=2 with guidance to avoid
    # "reshape tensor of 0 elements" bug in diffusers>=0.33
    _generate_all_pairs(
        pipe, pairs, out_dir,
        strength=0.8,  # SD-Turbo recommended strength
        num_inference_steps=2,
        device=device,
    )

    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------
def _generate_all_pairs(
    pipe,
    pairs: list,
    out_dir: Path,
    strength: float,
    device: str,
    num_inference_steps: int | None = None,
):
    total_pairs = len(pairs)

    for pair_idx, (src_style, tgt_style) in enumerate(pairs):
        desc = f"[{pair_idx+1}/{total_pairs}] {src_style} -> {tgt_style}"
        src_images = get_test_images(src_style)
        if not src_images:
            print(f"  {desc}: SKIP (no source images)")
            continue

        # Use up to NUM_IMAGES_PER_PAIR source images
        src_images = src_images[:NUM_IMAGES_PER_PAIR]

        for img_idx, src_path in enumerate(
            tqdm(src_images, desc=desc, leave=False)
        ):
            src_name = src_name_from_filename(src_path.name)
            out_name = f"{src_style}__{src_name}__to__{tgt_style}.png"
            out_path = out_dir / out_name

            if out_path.exists():
                continue

            seed = BASE_SEED + pair_idx * NUM_IMAGES_PER_PAIR + img_idx
            generator = torch.Generator(device="cpu").manual_seed(seed)

            try:
                init_image = Image.open(src_path).convert("RGB")
                init_image = init_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

                kwargs = dict(
                    image=init_image,
                    prompt="",
                    strength=strength,
                    guidance_scale=0.0,
                    generator=generator,
                )
                if num_inference_steps is not None:
                    kwargs["num_inference_steps"] = num_inference_steps

                result = pipe(**kwargs)
                img = result.images[0]
                img.save(str(out_path))

            except Exception as e:
                print(f"  ERROR: {out_name} -> {e}")
                continue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline images using SDEdit / SD-Turbo"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sdedit", "sdturbo", "all"],
        default="all",
        help="Which method to run (default: all)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs="*",
        default=None,
        help="SDEdit strength(s). Default: all four [0.10, 0.20, 0.35, 0.40]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Limit number of style pairs for quick testing (default: all 25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = build_pairs()
    if args.max_pairs is not None and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    print(f"Style pairs: {len(pairs)}")
    print(f"Images per pair: {NUM_IMAGES_PER_PAIR}")
    print(f"Output root: {OUTPUT_ROOT}")

    if args.method in ("sdedit", "all"):
        strengths = args.strength if args.strength else SDEDIT_STRENGTHS
        run_sdedit(strengths, pairs, args.device)

    if args.method in ("sdturbo", "all"):
        run_sdturbo(pairs, args.device)

    print("\nDone.")


if __name__ == "__main__":
    main()
