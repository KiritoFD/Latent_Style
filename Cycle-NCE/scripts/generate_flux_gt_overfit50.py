from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image


STYLE_PROMPTS = {
    "hayao": "Studio Ghibli style, Hayao Miyazaki inspired, painterly anime background, hand-painted texture, cinematic light",
    "monet": "Claude Monet impressionist oil painting, soft natural light, rich brush strokes, color harmony",
    "vangogh": "Vincent van Gogh expressive oil painting, swirling brush strokes, vivid colors, impasto texture",
    "cezanne": "Paul Cezanne post-impressionist painting, geometric brush planes, structured color blocks, oil paint texture",
    "photo": "High quality real life photography, DSLR, ultra realistic, sharp focus, natural lighting",
}


def resolve_path(raw: str | Path, bases: list[Path]) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    for b in bases:
        c = (b / p).resolve()
        if c.exists():
            return c
    return (bases[0] / p).resolve()


def load_styles_from_config(config_path: Path) -> tuple[list[str], Path]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    styles = [str(x) for x in cfg.get("data", {}).get("style_subdirs", [])]
    test_image_dir = str(cfg.get("training", {}).get("test_image_dir", ""))
    return styles, Path(test_image_dir)


def build_pipeline(
    model_id: str,
    dtype: torch.dtype,
    device: str,
    use_4bit: bool,
    use_controlnet_canny: bool,
    controlnet_id: str,
):
    if str(model_id).lower().startswith("ms://"):
        ms_model_id = str(model_id)[5:]
        from modelscope.hub.snapshot_download import snapshot_download

        ms_token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip() or None
        local_model_dir = snapshot_download(model_id=ms_model_id, token=ms_token)
        model_id = local_model_dir

    if use_controlnet_canny:
        from diffusers import FluxControlNetModel, FluxControlNetPipeline
        from transformers import BitsAndBytesConfig

        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            )

        controlnet = FluxControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        pipe = FluxControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
        )
        canny = None
        from controlnet_aux import CannyDetector

        canny = CannyDetector()
    else:
        from diffusers import FluxImg2ImgPipeline
        from transformers import BitsAndBytesConfig

        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            )
        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
        )
        canny = None

    if hasattr(pipe, "enable_model_cpu_offload") and device.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe, canny


def style_prompt(style_name: str) -> str:
    s = style_name.strip().lower()
    if s in STYLE_PROMPTS:
        return STYLE_PROMPTS[s]
    return f"{style_name} style painting, high quality artistic rendering"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLUX-based 5x5 GT matrix for overfit50 with low-VRAM settings.")
    parser.add_argument("--config", type=str, default="Cycle-NCE/src/config.json")
    parser.add_argument("--src_root", type=str, default="", help="Source root directory. Default: <test_image_dir> from config.")
    parser.add_argument("--out_root", type=str, default="style_data/flux-gt")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--use_4bit", action="store_true", help="Enable bitsandbytes 4-bit NF4 quantization.")
    parser.add_argument("--use_controlnet_canny", action="store_true", help="Enable FLUX ControlNet Canny path.")
    parser.add_argument("--controlnet_id", type=str, default="promeai/FLUX.1-controlnet-canny")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.65)
    parser.add_argument("--styles", type=str, default="", help="Comma-separated styles for both source/target. Default: config styles.")
    parser.add_argument("--identity_mode", type=str, choices=["copy", "flux"], default="copy", help="How to handle src==tgt pairs.")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--downsample_to", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Max number of source images per style; <=0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    # Prefer repository root (Latent_Style) regardless of current working directory.
    bases = [here.parents[2], Path.cwd(), here.parents[1], here.parent]
    config_path = resolve_path(args.config, bases)
    out_arg = Path(args.out_root).expanduser()
    out_root = out_arg.resolve() if out_arg.is_absolute() else (bases[0] / out_arg).resolve()

    styles_from_cfg, test_image_dir_raw = load_styles_from_config(config_path)
    test_image_dir = resolve_path(test_image_dir_raw, [config_path.parent, *bases])
    src_root = resolve_path(args.src_root, [config_path.parent, *bases]) if str(args.src_root).strip() else test_image_dir
    if not src_root.exists():
        raise SystemExit(f"Source root dir not found: {src_root}")

    if args.styles.strip():
        styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    else:
        styles = list(styles_from_cfg)

    src_images_by_style: dict[str, list[Path]] = {}
    for s in styles:
        s_dir = src_root / s
        imgs = []
        if s_dir.exists():
            imgs = sorted(
                p for p in s_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        if args.limit > 0:
            imgs = imgs[: args.limit]
        src_images_by_style[s] = imgs
    n_src = sum(len(v) for v in src_images_by_style.values())

    print(f"[CONFIG] {config_path}")
    print(f"[SRC_ROOT] {src_root}")
    print(f"[OUT_ROOT] {out_root}")
    print(f"[MODEL] {args.model_id}")
    print(f"[CONTROLNET] {args.use_controlnet_canny} ({args.controlnet_id})")
    print(f"[4BIT] {args.use_4bit}")
    print(f"[STYLES] {styles}")
    print(f"[N_SRC_TOTAL] {n_src}")

    plans: list[tuple[str, Path, str, Path]] = []
    for src_style in styles:
        for src in src_images_by_style.get(src_style, []):
            safe_stem = src.stem.replace(" ", "_")
            for tgt_style in styles:
                out_name = f"{src_style}_{safe_stem}_to_{tgt_style}.jpg"
                dst = out_root / "images" / out_name
                plans.append((src_style, src, tgt_style, dst))

    if args.dry_run:
        print(f"[DRY_RUN] planned={len(plans)}")
        for src_style, src, tgt_style, dst in plans[:20]:
            print(f"  {src_style}:{src.name} -> {tgt_style} -> {dst}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.startswith("cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    pipe, canny_detector = build_pipeline(
        args.model_id,
        dtype=dtype,
        device=device,
        use_4bit=bool(args.use_4bit),
        use_controlnet_canny=bool(args.use_controlnet_canny),
        controlnet_id=str(args.controlnet_id),
    )
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(int(args.seed))

    done = 0
    skip = 0
    fail = 0
    for src_style, src_path, tgt_style, dst_path in plans:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and not args.overwrite:
            skip += 1
            continue
        if src_style.lower() == tgt_style.lower() and args.identity_mode == "copy":
            img = Image.open(src_path).convert("RGB")
            if int(args.downsample_to) > 0:
                img = img.resize((int(args.downsample_to), int(args.downsample_to)), Image.Resampling.LANCZOS)
            img.save(dst_path)
            done += 1
            continue

        init_img = Image.open(src_path).convert("RGB").resize((int(args.width), int(args.height)), Image.Resampling.LANCZOS)
        prompt = (
            f"{style_prompt(tgt_style)}. Preserve scene composition. "
            f"High detail, clean brush strokes, coherent global lighting."
        )
        if args.use_controlnet_canny:
            control_image = canny_detector(init_img, low_threshold=100, high_threshold=200)
            out = pipe(
                prompt=prompt,
                control_image=control_image,
                controlnet_conditioning_scale=float(args.controlnet_conditioning_scale),
                guidance_scale=float(args.guidance_scale),
                num_inference_steps=int(args.num_inference_steps),
                generator=generator,
            ).images[0]
        else:
            out = pipe(
                prompt=prompt,
                image=init_img,
                strength=float(args.strength),
                guidance_scale=float(args.guidance_scale),
                num_inference_steps=int(args.num_inference_steps),
                generator=generator,
            ).images[0]

        if int(args.downsample_to) > 0:
            out = out.resize((int(args.downsample_to), int(args.downsample_to)), Image.Resampling.LANCZOS)
        out.save(dst_path)
        done += 1
        if done % 10 == 0:
            print(f"[PROGRESS] done={done} skip={skip} fail={fail}")

    print(f"[DONE] generated={done} skipped={skip} failed={fail} out_root={out_root}")


if __name__ == "__main__":
    main()
