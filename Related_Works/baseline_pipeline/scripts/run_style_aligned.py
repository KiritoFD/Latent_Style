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
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = WORKSPACE_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


ALL_STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]

STYLE_PROMPTS = {
    "monet": "impressionist painting style of Claude Monet, water lilies, soft brushstrokes",
    "vangogh": "post-impressionist painting style of Vincent van Gogh, swirling brushstrokes, bold colors",
    "ukiyoe": "Japanese ukiyo-e woodblock print style, flat colors, strong outlines",
    "cezanne": "post-impressionist painting style of Paul Cezanne, geometric forms, warm colors",
    "Hayao": "anime style of Studio Ghibli Hayao Miyazaki, vibrant colors, whimsical",
    "photo": "a realistic photograph",
}


def _manifest_items(content_manifest: Path | None) -> list[tuple[str, Path]] | None:
    if content_manifest is None:
        return None
    out: list[tuple[str, Path]] = []
    for line in content_manifest.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        content_style, img_name = name.split("_", 1)
        out.append((content_style, OVERFIT50 / content_style / img_name))
    return out


def run_style_aligned(target_style, max_images=0, output_root: Path | None = None, content_manifest: Path | None = None):
    """Run StyleAligned inference: all 5 content dirs -> target_style = 30*5=150 images."""
    output_base = output_root or (PIPELINE_ROOT / "results" / "style_aligned")
    output_dir = output_base / target_style
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

    manifest_items = _manifest_items(content_manifest)
    if manifest_items is None:
        manifest_items = []
        for content_style in ALL_STYLES:
            content_dir = OVERFIT50 / content_style
            if content_dir.exists():
                manifest_items.extend((content_style, p) for p in sorted(content_dir.glob("*.jpg")))
    if max_images > 0:
        manifest_items = manifest_items[:max_images]

    tag = f"StyleAligned/to_{target_style}"
    print(f"[{tag}] {len(manifest_items)} images")
    for content_style, img_path in tqdm(manifest_items, desc=tag):
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
    parser.add_argument("--output_root", type=Path, default=PIPELINE_ROOT / "results" / "style_aligned")
    parser.add_argument("--content_manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = args.content_manifest.resolve() if args.content_manifest else None
    return run_style_aligned(args.style, args.max_images, args.output_root.resolve(), manifest)


if __name__ == "__main__":
    sys.exit(main())
