#!/usr/bin/env python3
"""Unified StyleID / SD-Turbo baseline generator (training-free diffusion img2img).

Works for both WikiArt-20 (512) and legacy256 (5 styles, 256).
Skip-resumable: existing output PNGs are skipped.

Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
Output structure: {output_dir}/*.png  +  {output_dir}/../_DONE

Usage:
  # StyleID on wiki20 (512)
  python _gen_diffusion_baseline.py --method styleid \
    --test-dir I:\datasets\wikiarts20_512_test \
    --output-dir I:\...\exp\baseline_wikiarts20\styleid\images \
    --styles "Abstract_Expressionism,...,Ukiyo_e" --image-size 512

  # SD-Turbo on legacy256
  python _gen_diffusion_baseline.py --method sdturbo \
    --test-dir I:\datasets\legacy256_overfit50\test \
    --output-dir I:\exp_256_photo2art\sdturbo_256\images \
    --styles "cezanne,Hayao,monet,photo,vangogh" --image-size 256
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

# Force offline mode BEFORE importing diffusers/torch (remote has no internet;
# all required models are pre-cached in ~/.cache/huggingface/hub).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Style prompts ──
WIKI20_PROMPTS = {
    "Abstract_Expressionism": "a painting in Abstract Expressionism style, bold gestural brushstrokes",
    "Art_Nouveau_Modern": "a painting in Art Nouveau style, ornamental decorative lines",
    "Baroque": "a painting in Baroque style, dramatic chiaroscuro lighting",
    "Color_Field_Painting": "a painting in Color Field style, large flat areas of color",
    "Cubism": "a painting in Cubism style, geometric fragmented forms",
    "Early_Renaissance": "a painting in Early Renaissance style, tempera on panel",
    "Expressionism": "a painting in Expressionism style, vivid emotional colors",
    "Fauvism": "a painting in Fauvism style, wild unnatural vibrant colors",
    "High_Renaissance": "a painting in High Renaissance style, balanced classical composition",
    "Impressionism": "a painting in Impressionism style, soft brushstrokes light and color",
    "Mannerism_Late_Renaissance": "a painting in Mannerism style, elongated figures complex poses",
    "Minimalism": "a painting in Minimalism style, simple geometric forms",
    "Naive_Art_Primitivism": "a painting in Naive Art Primitivism style, simple childlike forms",
    "Northern_Renaissance": "a painting in Northern Renaissance style, detailed realistic oil technique",
    "Pop_Art": "a painting in Pop Art style, bold commercial imagery",
    "Post_Impressionism": "a painting in Post-Impressionism style, structured brushwork vivid color",
    "Rococo": "a painting in Rococo style, ornate decorative pastel colors",
    "Romanticism": "a painting in Romanticism style, dramatic emotional sublime scenery",
    "Symbolism": "a painting in Symbolism style, dreamlike metaphorical imagery",
    "Ukiyo_e": "a painting in Ukiyo-e style, Japanese woodblock print flat colors strong outlines",
}

LEGACY256_PROMPTS = {
    "cezanne": "a painting in Paul Cezanne style, geometric forms",
    "Hayao": "anime art by Hayao Miyazaki, Studio Ghibli style",
    "monet": "a painting in Claude Monet style, soft brushstrokes water lilies",
    "photo": "a realistic photograph",
    "vangogh": "a painting in Vincent van Gogh style, swirling bold brushstrokes",
}

METHOD_CONFIG = {
    "styleid": {
        "model": "runwayml/stable-diffusion-v1-5",
        "strength": 0.65,
        "steps": 50,
        "guidance": 7.5,
        "negative": "ugly, blurry, low quality, distorted",
    },
    "sdturbo": {
        "model": "stabilityai/sd-turbo",
        # SD-Turbo: img2img needs int(strength*steps) >= 1.
        # steps=4, strength=0.5 => 2 timesteps (good balance of speed & content preservation)
        "strength": 0.5,
        "steps": 4,
        "guidance": 1.0,
        "negative": "",
    },
}


def get_prompts(styles, test_dir):
    """Return prompt dict for the given styles."""
    prompts = {}
    for s in styles:
        if s in WIKI20_PROMPTS:
            prompts[s] = WIKI20_PROMPTS[s]
        elif s in LEGACY256_PROMPTS:
            prompts[s] = LEGACY256_PROMPTS[s]
        else:
            prompts[s] = f"a painting in {s.replace('_', ' ')} style"
    return prompts


def collect_sources(test_dir, styles, max_per_style, seed=42):
    rng = random.Random(seed)
    sources = []
    for style in styles:
        style_dir = test_dir / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found, skipping", flush=True)
            continue
        imgs = sorted(p for p in style_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(imgs)
        if max_per_style > 0:
            imgs = imgs[:max_per_style]
        for p in imgs:
            sources.append((style, p))
    return sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["styleid", "sdturbo"], required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--styles", type=str, required=True,
                   help="Comma-separated style names")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--max-src-per-style", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles:
        print("[ERROR] No styles provided", flush=True)
        return 1

    cfg = METHOD_CONFIG[args.method]
    prompts = get_prompts(styles, args.test_dir)

    print(f"=== {args.method} baseline generation ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  test_dir: {args.test_dir}", flush=True)
    print(f"  output_dir: {args.output_dir}", flush=True)
    print(f"  styles({len(styles)}): {styles}", flush=True)
    print(f"  image_size: {args.image_size}", flush=True)
    print(f"  model: {cfg['model']}", flush=True)
    print(f"  strength={cfg['strength']} steps={cfg['steps']} guidance={cfg['guidance']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(args.test_dir, styles, args.max_src_per_style, args.seed)
    total = len(sources) * len(styles)
    print(f"  {len(sources)} srcs x {len(styles)} styles = {total} images", flush=True)

    # Count existing
    existing = len(list(args.output_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)
    if existing >= total:
        print(f"  All images exist, skipping generation.", flush=True)
        done = args.output_dir.parent / "_DONE"
        done.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        return 0

    # Load pipeline (use local_files_only=True since remote has no internet;
    # models are pre-cached in ~/.cache/huggingface/hub)
    print(f"  Loading {cfg['model']} (local_files_only=True)...", flush=True)
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg["model"], torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    n_new = 0
    n_skip = 0
    t0 = time.time()

    for src_style, src_path in sources:
        src_stem = src_path.stem
        for tgt_style in styles:
            out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
            out_path = args.output_dir / out_name
            if out_path.exists():
                n_skip += 1
                continue
            try:
                content_img = Image.open(src_path).convert("RGB").resize(
                    (args.image_size, args.image_size), Image.LANCZOS)
                prompt = prompts.get(tgt_style, f"a painting in {tgt_style} style")
                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=cfg["negative"] if cfg["negative"] else None,
                        image=content_img,
                        strength=cfg["strength"],
                        num_inference_steps=cfg["steps"],
                        guidance_scale=cfg["guidance"],
                    ).images[0]
                result.save(out_path)
                n_new += 1
            except Exception as e:
                print(f"[WARN] Failed {src_style}->{tgt_style} ({src_stem}): {e}", flush=True)

            if (n_new + n_skip) % 50 == 0:
                elapsed = time.time() - t0
                rate = (n_new + n_skip) / max(elapsed, 1)
                eta = (total - n_new - n_skip) / max(rate, 0.01)
                print(f"  progress: {n_new + n_skip}/{total}  new={n_new} skip={n_skip}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {n_new} new + {n_skip} skipped in {elapsed:.1f}s", flush=True)

    done = args.output_dir.parent / "_DONE"
    done.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
