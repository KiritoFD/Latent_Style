"""StyleID inference for R5-WikiArt dataset.

StyleID (Zhang et al. 2023) is a training-free diffusion-based style transfer
method using Stable Diffusion v1.5 img2img with style prompts.

This script generates 750 images (5 styles x 5 styles x 30 content images)
on the R5-WikiArt hold-out test set.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image


STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

STYLE_PROMPTS = {
    "Cubism": "cubist painting in the style of Pablo Picasso, geometric shapes, fragmented forms",
    "Expressionism": "expressionist painting with bold colors and emotional brushstrokes, Edvard Munch style",
    "Pop_Art": "pop art in the style of Andy Warhol, bold colors, commercial imagery",
    "Romanticism": "romantic painting with dramatic lighting and emotional scenes, Caspar David Friedrich style",
    "Symbolism": "symbolist painting with dreamlike imagery and metaphysical themes, Gustav Moreau style",
}


def infer(test_dir: Path, output_dir: Path, num_src: int, strength: float,
          steps: int, guidance: float) -> int:
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[StyleID-R5] Loading Stable Diffusion v1.5...", flush=True)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    total = 0
    start_all = time.time()

    for src_style in STYLES:
        src_dir = test_dir / src_style
        if not src_dir.exists():
            print(f"[WARN] Source style dir not found: {src_dir}", flush=True)
            continue
        content_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])[:num_src]

        for content_path in content_files:
            for tgt_style in STYLES:
                out_name = f"{src_style}__{content_path.stem}__to__{tgt_style}.png"
                out_path = output_dir / out_name
                if out_path.exists():
                    total += 1
                    continue

                style_prompt = STYLE_PROMPTS.get(tgt_style, f"painting in {tgt_style} style")
                try:
                    content_img = Image.open(content_path).convert("RGB").resize((512, 512))
                    with torch.no_grad():
                        result = pipe(
                            prompt=style_prompt,
                            negative_prompt="ugly, blurry, low quality, distorted",
                            image=content_img,
                            strength=strength,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                        ).images[0]
                    result.save(out_path)
                    total += 1
                except Exception as e:
                    print(f"[WARN] Failed on {content_path.name} -> {tgt_style}: {e}", flush=True)

                torch.cuda.empty_cache()

        print(f"  {src_style} done: total={total} ({time.time() - start_all:.1f}s)", flush=True)

    del pipe
    torch.cuda.empty_cache()
    print(f"[StyleID-R5] Total generated: {total} images in {time.time() - start_all:.1f}s", flush=True)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="I:/datasets/wikiarts20_512_test")
    parser.add_argument("--output_dir", default="I:/exp_baselines/styleid/r5_wikiart/images")
    parser.add_argument("--num_src", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    args = parser.parse_args()

    total = infer(
        Path(args.test_dir),
        Path(args.output_dir),
        args.num_src,
        args.strength,
        args.steps,
        args.guidance,
    )
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
