"""Run StyleAligned (SD1.5) on Distinct5-WikiArt for Figure 1 baseline.

Outputs follow the baseline bridge naming convention:
    {src_style}__{src_stem}__to__{tgt_style}.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "style_aligned"))
from sa_handler_sd15 import Handler, StyleAlignedArgs
import inversion_sd15 as inversion


STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_PROMPT_SUFFIX = {
    "Early_Renaissance": "early renaissance painting",
    "Impressionism": "impressionist painting",
    "Minimalism": "minimalist painting",
    "Rococo": "rococo painting",
    "Ukiyo_e": "ukiyo-e painting",
}
DEFAULT_TEST_MANIFEST = Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test_manifest.json")
DEFAULT_OUT_ROOT = Path("G:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned_distinct5")


def pick_style_reference(test_dir: Path, style: str) -> Image.Image:
    style_dir = test_dir / style
    candidates = sorted([p for p in style_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not candidates:
        raise FileNotFoundError(f"No reference images found for style {style} in {style_dir}")
    img = Image.open(candidates[0]).convert("RGB").resize((512, 512), Image.LANCZOS)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--inversion_guidance_scale", type=float, default=3.5)
    parser.add_argument("--inversion_offset", type=int, default=0)
    parser.add_argument("--shared_score_shift", type=float, default=0.0)
    parser.add_argument("--shared_score_scale", type=float, default=1.0)
    parser.add_argument("--only_self_level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--src_style", type=str, default="", help="Only generate transfers for this source style (useful for chunking).")
    args = parser.parse_args()

    out_dir = args.out_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.test_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_dir = Path(manifest["test_dir"])
    style_files = manifest["style_files"]

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"Loading SD1.5 pipeline from {args.sd_model} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    handler = Handler(pipe)
    sa_args = StyleAlignedArgs(
        share_group_norm=True,
        share_layer_norm=True,
        share_attention=True,
        adain_queries=True,
        adain_keys=True,
        adain_values=False,
        shared_score_shift=args.shared_score_shift,
        shared_score_scale=args.shared_score_scale,
        only_self_level=args.only_self_level,
    )
    handler.register(sa_args)

    # Precompute style-reference inversions (one per target style).
    print("Precomputing DDIM inversions for style references ...")
    style_refs = {s: pick_style_reference(test_dir, s) for s in STYLES}
    style_inversions: dict[str, torch.Tensor] = {}
    for style in STYLES:
        ref_prompt = f"a {STYLE_PROMPT_SUFFIX[style]}"
        zts = inversion.ddim_inversion(
            pipe,
            style_refs[style],
            ref_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.inversion_guidance_scale,
        )
        style_inversions[style] = zts
        print(f"  Inversion done for {style}: zts shape {zts.shape}")

    src_styles = [args.src_style] if args.src_style else STYLES
    if args.src_style and args.src_style not in STYLES:
        raise ValueError(f"Unknown src_style {args.src_style}; choose from {STYLES}")

    pairs: list[tuple[str, str, str]] = []
    for src_style in src_styles:
        for tgt_style in STYLES:
            for fname in style_files[src_style]:
                src_stem = Path(fname).stem
                pairs.append((src_style, src_stem, tgt_style))

    total = len(pairs)
    print(f"Total pairs to generate: {total}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    timing_log = []
    start_all = time.time()

    for idx, (src_style, src_stem, tgt_style) in enumerate(pairs, 1):
        out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
        out_path = out_dir / out_name
        if args.resume and out_path.exists():
            continue

        src_path = test_dir / src_style / f"{src_stem}.jpg"
        if not src_path.exists():
            src_path = test_dir / src_style / f"{src_stem}.png"
        src_img = Image.open(src_path).convert("RGB").resize((512, 512), Image.LANCZOS)

        ref_prompt = f"a {STYLE_PROMPT_SUFFIX[tgt_style]}"
        content_prompt = ref_prompt

        zts = style_inversions[tgt_style]
        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=args.inversion_offset)

        latents = torch.randn(
            2,  # [reference, content]
            4,
            64,
            64,
            device="cpu",
            generator=generator,
            dtype=pipe.unet.dtype,
        ).to(args.device)
        latents[0] = zT

        t0 = time.time()
        images = pipe(
            [ref_prompt, content_prompt],
            latents=latents,
            callback_on_step_end=inversion_callback,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images
        dt = time.time() - t0

        # images[1] is the content image with style aligned to reference.
        images[1].save(out_path)
        timing_log.append((out_name, dt))

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start_all
            eta = elapsed / idx * (total - idx) if idx else 0
            print(f"[{idx}/{total}] {out_name}  dt={dt:.2f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    handler.remove()

    total_time = time.time() - start_all
    print(f"Done. Total time: {total_time/60:.2f} min ({total_time/total:.2f} s/image)")

    meta = {
        "method": "style_aligned_sd15_transfer",
        "test_manifest": str(args.test_manifest),
        "out_dir": str(out_dir),
        "sd_model": args.sd_model,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "inversion_guidance_scale": args.inversion_guidance_scale,
        "inversion_offset": args.inversion_offset,
        "shared_score_shift": args.shared_score_shift,
        "shared_score_scale": args.shared_score_scale,
        "only_self_level": args.only_self_level,
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
