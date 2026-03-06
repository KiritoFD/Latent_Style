import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


DEFAULT_PROMPTS = {
    "photo": "a realistic photograph",
    "monet": "an impressionist oil painting in Claude Monet style",
    "vangogh": "an expressive painting in Vincent van Gogh style",
    "cezanne": "a post-impressionist painting in Paul Cezanne style",
    "Hayao": "a hand-drawn anime frame in Hayao Miyazaki style",
}


def parse_prompt_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --prompt_override: {item}, expected style=text")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid --prompt_override: {item}")
        overrides[k] = v
    return overrides


def list_dataset_images(root: Path):
    styles = sorted([d.name for d in root.iterdir() if d.is_dir()], key=lambda x: x.lower())
    src = []
    for s in styles:
        for p in sorted((root / s).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                src.append((s, p))
    return styles, src


def main():
    ap = argparse.ArgumentParser("Generate 5x5 SDEdit outputs (256x256)")
    ap.add_argument("--test_dir", required=True, help="Dataset root containing style subfolders")
    ap.add_argument("--out_dir", required=True, help="Output root. Images go to out_dir/images")
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--size", type=int, default=256, help="Final output size (square)")
    ap.add_argument("--strength", type=float, default=0.35, help="Lower is more content-preserving")
    ap.add_argument("--steps", type=int, default=30, help="Sampling steps for img2img")
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_identity", action="store_true", help="If set, do not copy src->same-style")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    ap.add_argument("--max_per_style", type=int, default=0, help="0 means all")
    ap.add_argument("--prompt_override", action="append", default=[], help="style=prompt text")
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    overrides = parse_prompt_overrides(args.prompt_override)
    prompts = dict(DEFAULT_PROMPTS)
    prompts.update(overrides)

    styles, all_src = list_dataset_images(test_dir)
    if args.max_per_style > 0:
        grouped = {}
        for s, p in all_src:
            grouped.setdefault(s, []).append((s, p))
        all_src = []
        for s in styles:
            all_src.extend(grouped.get(s, [])[: args.max_per_style])

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    total = len(all_src) * len(styles)
    done = 0
    started = time.time()

    for src_style, src_path in all_src:
        src_img = Image.open(src_path).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
        for tgt_style in styles:
            out_name = f"{src_style}_{src_path.stem}_to_{tgt_style}.jpg"
            out_path = img_dir / out_name
            if out_path.exists() and not args.overwrite:
                done += 1
                continue

            if src_style == tgt_style and not args.skip_identity:
                src_img.save(out_path, quality=95)
            else:
                prompt = prompts.get(tgt_style, f"an artwork in {tgt_style} style")
                img = pipe(
                    prompt=prompt,
                    image=src_img,
                    strength=float(args.strength),
                    num_inference_steps=int(args.steps),
                    guidance_scale=float(args.guidance),
                    generator=gen,
                ).images[0]
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
                img.save(out_path, quality=95)

            done += 1
            if done % 20 == 0:
                mins = (time.time() - started) / 60.0
                print(f"[sdedit] {done}/{total} ({mins:.1f} min)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    meta = {
        "method": "sdedit",
        "model_id": args.model_id,
        "size": args.size,
        "strength": args.strength,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "styles": styles,
        "source_count": len(all_src),
        "prompts": prompts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sdedit] done -> {img_dir}")


if __name__ == "__main__":
    main()
