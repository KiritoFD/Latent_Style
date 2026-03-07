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


def _parse_strengths(text: str) -> list[float]:
    parts = [p.strip() for p in str(text).split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("Empty --strengths. Example: --strengths 0.3,0.6,0.8")
    return vals


def _enable_low_vram(pipe, device: str):
    if device != "cuda":
        return
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    # Prefer the most aggressive offload, fallback to model_cpu_offload.
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pipe.enable_model_cpu_offload()

def _is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


def _run_img2img_batched(
    pipe: StableDiffusionImg2ImgPipeline,
    *,
    prompts: list[str],
    image: Image.Image,
    strength: float,
    steps: int,
    guidance: float,
    generator: torch.Generator,
    batch_size: int,
):
    """
    Run img2img in batches to improve throughput.
    If CUDA OOM happens, automatically halves batch size and retries.
    """
    out: list[Image.Image] = []
    idx = 0
    cur_bs = max(1, int(batch_size))
    while idx < len(prompts):
        chunk_prompts = prompts[idx : idx + cur_bs]
        chunk_images = [image] * len(chunk_prompts)
        try:
            with torch.inference_mode():
                res = pipe(
                    prompt=chunk_prompts,
                    image=chunk_images,
                    strength=float(strength),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    generator=generator,
                )
            out.extend(list(res.images))
            idx += len(chunk_prompts)
        except RuntimeError as e:
            if torch.cuda.is_available() and _is_cuda_oom(e) and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                torch.cuda.empty_cache()
                continue
            raise
    return out


def main():
    ap = argparse.ArgumentParser("Generate 5x5 SDEdit outputs (256x256, batch-friendly)")
    ap.add_argument("--test_dir", required=True, help="Dataset root containing style subfolders")
    ap.add_argument("--out_dir", required=True, help="Output root. Subfolders per strength: out_dir/str_0.60/images")
    ap.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--size", type=int, default=256, help="Final output size (square)")
    ap.add_argument("--strengths", type=str, default="0.3,0.45,0.6,0.8", help="Comma-separated strengths, e.g. 0.3,0.6,0.8")
    ap.add_argument("--steps", type=int, default=50, help="Sampling steps for img2img")
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_identity", action="store_true", help="If set, do not copy src->same-style")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    ap.add_argument("--max_per_style", type=int, default=0, help="0 means all")
    ap.add_argument("--prompt_override", action="append", default=[], help="style=prompt text")
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4, help="Img2img batch size (CUDA). Increase to use more VRAM.")
    ap.add_argument("--low_vram", action="store_true", help="Enable attention slicing + VAE slicing + CPU offload (slower)")
    ap.add_argument("--empty_cache_every", type=int, default=0, help="Call torch.cuda.empty_cache() every N outputs; 0 disables")
    args = ap.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    strengths = _parse_strengths(args.strengths)

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
    print(f"[sdedit] Loading {args.model_id} dtype={dtype} device={device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if device == "cuda":
        if args.low_vram:
            _enable_low_vram(pipe, device)
        else:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Generator on CPU is more stable when using sequential offload.
    gen_dev = "cpu" if device == "cuda" else device
    gen = torch.Generator(device=gen_dev)
    gen.manual_seed(int(args.seed))

    per_strength = []
    total_started = time.perf_counter()
    total_images_per_strength = len(all_src) * len(styles)

    for s in strengths:
        s_dir = out_dir / f"str_{s:.2f}"
        img_dir = s_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        gen_count = 0
        infer_time_sec = 0.0
        started = time.perf_counter()

        cache_every = int(args.empty_cache_every)
        for src_style, src_path in all_src:
            src_img = Image.open(src_path).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
            # 1) Identity copies (cheap, no diffusion).
            pending: list[tuple[str, Path]] = []
            pending_prompts: list[str] = []
            for tgt_style in styles:
                out_name = f"{src_style}_{src_path.stem}_to_{tgt_style}.jpg"
                out_path = img_dir / out_name
                if out_path.exists() and not args.overwrite:
                    done += 1
                    continue
                if src_style == tgt_style and not args.skip_identity:
                    src_img.save(out_path, quality=95)
                    done += 1
                    continue
                prompt = prompts.get(tgt_style, f"an artwork in {tgt_style} style")
                pending.append((tgt_style, out_path))
                pending_prompts.append(prompt)

            # 2) Batched diffusion for non-identity targets.
            if pending_prompts:
                t0 = time.perf_counter()
                imgs = _run_img2img_batched(
                    pipe,
                    prompts=pending_prompts,
                    image=src_img,
                    strength=float(s),
                    steps=int(args.steps),
                    guidance=float(args.guidance),
                    generator=gen,
                    batch_size=int(args.batch),
                )
                infer_time_sec += time.perf_counter() - t0
                for (_, out_path), img in zip(pending, imgs):
                    gen_count += 1
                    img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
                    img.save(out_path, quality=95)
                    done += 1

            if args.progress_every > 0 and done % int(args.progress_every) == 0:
                mins = (time.perf_counter() - started) / 60.0
                print(f"[sdedit str={s:.2f}] {done}/{total_images_per_strength} ({mins:.1f} min)")
            if torch.cuda.is_available() and cache_every > 0 and done % cache_every == 0:
                torch.cuda.empty_cache()

        elapsed = time.perf_counter() - started
        meta = {
            "method": "sdedit",
            "model_id": args.model_id,
            "size": args.size,
            "strength": float(s),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "seed": int(args.seed),
            "batch": int(args.batch),
            "low_vram": bool(args.low_vram),
            "empty_cache_every": int(args.empty_cache_every),
            "styles": styles,
            "source_count": len(all_src),
            "prompts": prompts,
            "generated_images": int(gen_count),
            "elapsed_sec": float(elapsed),
            "inference_sec": float(infer_time_sec),
            "avg_infer_sec_per_generated": float(infer_time_sec / max(gen_count, 1)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (s_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        per_strength.append(meta)
        print(f"[sdedit] strength {s:.2f} done -> {img_dir}")

    summary = {
        "method": "sdedit",
        "device": device,
        "dtype": str(dtype),
        "model_id": args.model_id,
        "size": args.size,
        "strengths": strengths,
        "steps": int(args.steps),
        "guidance": float(args.guidance),
        "seed": int(args.seed),
        "styles": styles,
        "source_count": len(all_src),
        "images_per_strength": int(total_images_per_strength),
        "runs": per_strength,
        "total_elapsed_sec": float(time.perf_counter() - total_started),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sdedit] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
