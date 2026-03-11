import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
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


def _configure_memory(
    pipe,
    device: str,
    *,
    offload: str,
    attention_slicing: str,
    vae_slicing: bool,
):
    if device != "cuda":
        return pipe

    if attention_slicing and attention_slicing.lower() != "none":
        pipe.enable_attention_slicing(attention_slicing)
    if vae_slicing:
        pipe.enable_vae_slicing()

    off = str(offload or "").strip().lower()
    if off == "none":
        pipe = pipe.to("cuda")
    elif off == "model":
        pipe.enable_model_cpu_offload()
    elif off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"Invalid --offload: {offload}. Expected one of: none, model, sequential.")
    return pipe


def main():
    ap = argparse.ArgumentParser("Generate 5x5 SDXL-Turbo outputs (256x256, 6GB VRAM-friendly)")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="stabilityai/sdxl-turbo")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=1, help="Default 1. If fails, auto fallback to 2.")
    ap.add_argument("--strength", type=float, default=1.0, help="Use 1.0 for 1-step SDXL img2img.")
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_identity", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_per_style", type=int, default=0)
    ap.add_argument("--prompt_override", action="append", default=[])
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument(
        "--offload",
        choices=["sequential", "model", "none"],
        default="sequential",
        help="VRAM vs speed: sequential (min VRAM, slow), model (mid), none (fast, high VRAM).",
    )
    ap.add_argument(
        "--attention_slicing",
        choices=["max", "none"],
        default="max",
        help="Attention slicing reduces VRAM but slows inference.",
    )
    ap.add_argument(
        "--vae_slicing",
        action="store_true",
        help="Enable VAE slicing to reduce VRAM (slower).",
    )
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
    print(f"[sdturbo] Loading {args.model_id} dtype={dtype} device={device}")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    if device == "cuda":
        pipe = _configure_memory(
            pipe,
            device,
            offload=args.offload,
            attention_slicing=args.attention_slicing,
            vae_slicing=bool(args.vae_slicing),
        )
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    gen_dev = "cpu" if (device == "cuda" and str(args.offload).lower() != "none") else device
    gen = torch.Generator(device=gen_dev)
    gen.manual_seed(int(args.seed))

    total = len(all_src) * len(styles)
    done = 0
    started = time.perf_counter()
    gen_count = 0
    infer_time_sec = 0.0
    fallback_to_2 = 0

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
                try:
                    t0 = time.perf_counter()
                    img = pipe(
                        prompt=prompt,
                        image=src_img,
                        num_inference_steps=int(args.num_steps),
                        strength=float(args.strength),
                        guidance_scale=float(args.guidance),
                        generator=gen,
                    ).images[0]
                    infer_time_sec += time.perf_counter() - t0
                    gen_count += 1
                except RuntimeError:
                    fallback_to_2 += 1
                    t0 = time.perf_counter()
                    img = pipe(
                        prompt=prompt,
                        image=src_img,
                        num_inference_steps=max(2, int(args.num_steps)),
                        strength=float(args.strength),
                        guidance_scale=float(args.guidance),
                        generator=gen,
                    ).images[0]
                    infer_time_sec += time.perf_counter() - t0
                    gen_count += 1
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
                img.save(out_path, quality=95)

            done += 1
            if args.progress_every > 0 and done % int(args.progress_every) == 0:
                mins = (time.perf_counter() - started) / 60.0
                print(f"[sdturbo] {done}/{total} ({mins:.1f} min)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    meta = {
        "method": "sdturbo",
        "device": device,
        "dtype": str(dtype),
        "model_id": args.model_id,
        "size": args.size,
        "num_steps": args.num_steps,
        "strength": args.strength,
        "guidance": args.guidance,
        "seed": args.seed,
        "offload": args.offload,
        "attention_slicing": args.attention_slicing,
        "vae_slicing": bool(args.vae_slicing),
        "styles": styles,
        "source_count": len(all_src),
        "prompts": prompts,
        "generated_images": int(gen_count),
        "fallback_to_2_steps": int(fallback_to_2),
        "elapsed_sec": float(elapsed),
        "inference_sec": float(infer_time_sec),
        "avg_infer_sec_per_generated": float(infer_time_sec / max(gen_count, 1)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sdturbo] done -> {img_dir}")


if __name__ == "__main__":
    main()
