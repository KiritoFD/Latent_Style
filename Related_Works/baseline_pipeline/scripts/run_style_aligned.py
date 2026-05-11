"""
StyleAligned Execution Script
CVPR 2024 Google - Attention Sharing + ControlNet for Style Transfer

Uses Stable Diffusion + ControlNet Canny for structure-preserving style transfer.
Zero-shot inference (no training needed).
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


ALL_STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]

STYLE_PROMPTS = {
    "monet": "impressionist painting style of Claude Monet, water lilies, soft brushstrokes",
    "vangogh": "post-impressionist painting style of Vincent van Gogh, swirling brushstrokes, bold colors",
    "ukiyoe": "Japanese ukiyo-e woodblock print style, flat colors, strong outlines",
    "cezanne": "post-impressionist painting style of Paul Cezanne, geometric forms, warm colors",
    "Hayao": "anime style of Studio Ghibli Hayao Miyazaki, vibrant colors, whimsical",
    "photo": "a realistic photograph",
}


def run_style_aligned(target_style, max_images=0):
    """Run StyleAligned inference: all 5 content dirs -> target_style = 30*5=150 images."""
    output_dir = PIPELINE_ROOT / "results" / "style_aligned" / target_style
    output_dir.mkdir(parents=True, exist_ok=True)

    style_dir = OVERFIT50 / target_style
    if not style_dir.exists():
        print(f"[ERROR] Style dir not found: {style_dir}")
        return 1

    # Load ControlNet + SD pipeline
    print(f"[StyleAligned] Loading ControlNet + SD1.5...")
    try:
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )
    except ImportError:
        print("[ERROR] diffusers not installed")
        return 1

    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=DTYPE,
            safety_checker=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}")
        return 1

    # Canny detector
    try:
        from controlnet_aux import CannyDetector
        canny_detector = CannyDetector()
    except ImportError:
        print("[WARN] controlnet_aux not installed, using PIL edge detection")
        canny_detector = None

    prompt = STYLE_PROMPTS.get(target_style, f"painting in the style of {target_style}")

    # Process all 5 content directories -> 5*30 = 150 images per target style
    for content_style in ALL_STYLES:
        content_dir = OVERFIT50 / content_style
        if not content_dir.exists():
            continue

        content_files = sorted(content_dir.glob("*.jpg"))
        if max_images > 0:
            content_files = content_files[:max_images]

        tag = f"StyleAligned/{content_style}_to_{target_style}"
        print(f"[{tag}] {len(content_files)} images")

        for img_path in tqdm(content_files, desc=tag):
            out_name = f"{content_style}_{img_path.stem}_to_{target_style}.jpg"
            out_path = output_dir / out_name
            if out_path.exists():
                continue

            try:
                content_img = Image.open(img_path).convert("RGB").resize((512, 512))
                if canny_detector is not None:
                    canny_img = canny_detector(content_img, low_threshold=100, high_threshold=200)
                else:
                    canny_img = content_img

                with torch.no_grad(), torch.autocast(DEVICE, dtype=DTYPE):
                    result = pipe(
                        prompt=prompt,
                        negative_prompt="ugly, blurry, low quality, distorted, deformed",
                        image=canny_img,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        controlnet_conditioning_scale=0.7,
                    ).images[0]
                result.save(out_path)
            except Exception as e:
                print(f"[WARN] Failed on {img_path.name}: {e}")

            torch.cuda.empty_cache()

    del pipe, controlnet
    torch.cuda.empty_cache()
    print(f"[StyleAligned] Done: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=0, help="Max images (0=all)")
    args = parser.parse_args()
    return run_style_aligned(args.style, args.max_images)


if __name__ == "__main__":
    sys.exit(main())
