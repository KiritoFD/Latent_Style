"""Run IP-Adapter (SD1.5 img2img) on Distinct5-WikiArt for Figure 1 baseline.

Outputs follow the baseline bridge naming convention:
    {src_style}__{src_stem}__to__{tgt_style}.png
so they can be fed directly to src/utils/run_evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
DEFAULT_TEST_MANIFEST = Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test_manifest.json")
DEFAULT_OUT_ROOT = Path("G:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_ipadapter_distinct5")


def pick_style_reference(test_dir: Path, style: str) -> Image.Image:
    """Use the first image in the target-style test folder as the IP-Adapter reference."""
    style_dir = test_dir / style
    candidates = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not candidates:
        raise FileNotFoundError(f"No reference images found for style {style} in {style_dir}")
    img = Image.open(candidates[0]).convert("RGB")
    # IP-Adapter image encoder expects 224x224 internally; keep 512x512 input.
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.LANCZOS)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--ip_adapter_repo", type=str, default="h94/IP-Adapter")
    parser.add_argument("--ip_adapter_weight", type=str, default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--ip_adapter_subfolder", type=str, default="models")
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter conditioning scale.")
    parser.add_argument("--strength", type=float, default=0.65, help="img2img denoising strength.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--prompt", type=str, default="", help="Text prompt; empty by default for pure style transfer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist.")
    args = parser.parse_args()

    out_dir = args.out_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.test_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_dir = Path(manifest["test_dir"])
    style_files = manifest["style_files"]

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Loading SD1.5 img2img pipeline from {args.sd_model} ...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(args.device)
    pipe.load_ip_adapter(
        args.ip_adapter_repo,
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight,
    )
    pipe.set_ip_adapter_scale(args.scale)

    # Preload style references.
    style_refs = {s: pick_style_reference(test_dir, s) for s in STYLES}

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    pairs: list[tuple[str, str, str]] = []
    for src_style in STYLES:
        for tgt_style in STYLES:
            for fname in style_files[src_style]:
                src_stem = Path(fname).stem
                pairs.append((src_style, src_stem, tgt_style))

    total = len(pairs)
    print(f"Total pairs to generate: {total}")

    timing_log = []
    start_all = time.time()
    for idx, (src_style, src_stem, tgt_style) in enumerate(pairs, 1):
        out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
        out_path = out_dir / out_name
        if args.resume and out_path.exists():
            continue

        src_path = test_dir / src_style / f"{src_stem}.jpg"
        if not src_path.exists():
            # Fallback to png.
            src_path = test_dir / src_style / f"{src_stem}.png"

        src_img = Image.open(src_path).convert("RGB")
        if src_img.size != (512, 512):
            src_img = src_img.resize((512, 512), Image.LANCZOS)

        t0 = time.time()
        result = pipe(
            prompt=args.prompt,
            image=src_img,
            ip_adapter_image=style_refs[tgt_style],
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        dt = time.time() - t0

        result.save(out_path)
        timing_log.append((out_name, dt))
        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start_all
            eta = elapsed / idx * (total - idx) if idx else 0
            print(f"[{idx}/{total}] {out_name}  dt={dt:.2f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    total_time = time.time() - start_all
    print(f"Done. Total time: {total_time/60:.2f} min ({total_time/total:.2f} s/image)")

    # Save timing metadata.
    meta = {
        "method": "ip_adapter_plus_sd15_img2img",
        "test_manifest": str(args.test_manifest),
        "out_dir": str(out_dir),
        "sd_model": args.sd_model,
        "ip_adapter_weight": args.ip_adapter_weight,
        "scale": args.scale,
        "strength": args.strength,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "prompt": args.prompt,
        "seed": args.seed,
        "total_images": total,
        "total_seconds": total_time,
        "seconds_per_image": total_time / total if total else None,
        "per_image_timings": timing_log,
    }
    with (args.out_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
