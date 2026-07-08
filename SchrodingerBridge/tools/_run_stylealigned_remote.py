"""Run StyleAligned (SD1.5) on Photo2Art-256 and Random5-WikiArt on remote 3060.
D5 is already done locally (exp/baseline_stylealigned_distinct5).

This script must be scp'd to remote and run there.
Requires tools/style_aligned/ module to be uploaded alongside.
"""
import json, os, sys, time, gc
from pathlib import Path

# ── Add style_aligned module path ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "style_aligned"))

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

from sa_handler_sd15 import Handler, StyleAlignedArgs
import inversion_sd15 as inversion

SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16

# ── Dataset definitions ──
# Photo2Art-256: 5 styles, 256x256
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
P2A_SIZE = 256
P2A_STYLE_PROMPTS = {
    "cezanne": "a painting in Paul Cezanne style",
    "Hayao": "a painting in Hayao Miyazaki style",
    "monet": "a painting in Claude Monet style",
    "photo": "a photograph",
    "vangogh": "a painting in Vincent van Gogh style",
}

# Random5-WikiArt: 5 hold-out styles from wikiarts20, 512x512
R5_SIZE = 512
# Use 5 styles NOT in Distinct5 (Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e)
R5_HOLDOUT = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
R5_STYLE_PROMPTS = {
    "Cubism": "a painting in Cubism style",
    "Expressionism": "a painting in Expressionism style",
    "Pop_Art": "a painting in Pop Art style",
    "Romanticism": "a painting in Romanticism style",
    "Symbolism": "a painting in Symbolism style",
}

# Distinct5-WikiArt: 512x512
D5_SIZE = 512
D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
D5_STYLE_PROMPTS = {
    "Early_Renaissance": "an early renaissance painting",
    "Impressionism": "an impressionist painting",
    "Minimalism": "a minimalist painting",
    "Rococo": "a rococo painting",
    "Ukiyo_e": "a ukiyo-e painting",
}

def prompt_for(style: str, prompt_dict: dict) -> str:
    """Resolve a style prompt, falling back to a generic painting prompt for
    any style name not present in the explicit prompt dictionary (e.g. when the
    R5 hold-out set does not match the actual test-dir folder names)."""
    if style in prompt_dict:
        return prompt_dict[style]
    return f"a painting in {style.replace('_', ' ')} style"


OUT_ROOT = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline_stylealigned")


def scan_dir_for_images(dir_path: Path) -> list[Path]:
    """List all image files in a directory, sorted."""
    if not dir_path.exists():
        return []
    return sorted([f for f in dir_path.iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def load_pipeline():
    """Load SD1.5 pipeline + StyleAligned handler."""
    print("  Loading SD1.5 pipeline ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    handler = Handler(pipe)
    sa_args = StyleAlignedArgs(
        share_group_norm=True,
        share_layer_norm=True,
        share_attention=True,
        adain_queries=True,
        adain_keys=True,
        adain_values=False,
        shared_score_shift=0.0,
        shared_score_scale=1.0,
        only_self_level=0.0,
    )
    handler.register(sa_args)
    return pipe, handler


def run_inference(pipe, test_dir, style_list, style_prompts, out_dir,
                  img_size, dataset_label, inversion_steps=20,
                  guidance_scale=7.5, inv_guidance_scale=3.5):
    """Run StyleAligned inference on all style pairs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images per style
    sources_by_style = {}
    for style in style_list:
        style_dir = test_dir / style
        files = scan_dir_for_images(style_dir)
        if files:
            sources_by_style[style] = files

    # Pick first image per target style as reference
    style_ref_paths = {}
    for tgt_style in style_list:
        style_dir = test_dir / tgt_style
        files = scan_dir_for_images(style_dir)
        if files:
            style_ref_paths[tgt_style] = files[0]

    # Precompute DDIM inversions for style references
    print(f"  [{dataset_label}] Precomputing DDIM inversions for {len(style_ref_paths)} style refs ...", flush=True)
    style_inversions = {}
    for tgt_style, ref_path in style_ref_paths.items():
        ref_img = Image.open(ref_path).convert("RGB")
        # StyleAligned uses 512x512 internally for SD1.5
        if ref_img.size != (512, 512):
            ref_img = ref_img.resize((512, 512), Image.LANCZOS)
        ref_prompt = style_prompts[tgt_style]
        zts = inversion.ddim_inversion(
            pipe, ref_img, ref_prompt,
            num_inference_steps=inversion_steps,
            guidance_scale=inv_guidance_scale,
        )
        style_inversions[tgt_style] = zts
        print(f"    Inversion done: {tgt_style}, zts shape {zts.shape}", flush=True)

    # Generate all pairs
    all_pairs = []
    for src_style, files in sources_by_style.items():
        for src_path in files:
            src_stem = src_path.stem
            for tgt_style in style_list:
                if tgt_style in style_inversions:
                    all_pairs.append((src_style, src_stem, src_path, tgt_style))

    total = len(all_pairs)
    print(f"  [{dataset_label}] {total} pairs to generate", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    timing_log = []
    start_all = time.time()
    count = 0

    for idx, (src_style, src_stem, src_path, tgt_style) in enumerate(all_pairs, 1):
        out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
        out_path = out_dir / out_name
        if out_path.exists():
            count += 1
            continue

        src_img = Image.open(src_path).convert("RGB")
        # Resize to 512x512 for SD1.5 processing
        if src_img.size != (512, 512):
            src_img = src_img.resize((512, 512), Image.LANCZOS)

        ref_prompt = style_prompts[tgt_style]
        content_prompt = ref_prompt

        zts = style_inversions[tgt_style]
        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=0)

        latents = torch.randn(
            2, 4, 64, 64,
            device="cpu", generator=generator, dtype=pipe.unet.dtype,
        ).to(DEVICE)
        latents[0] = zT

        t0 = time.time()
        with torch.no_grad():
            images = pipe(
                [ref_prompt, content_prompt],
                latents=latents,
                callback_on_step_end=inversion_callback,
                num_inference_steps=inversion_steps,
                guidance_scale=guidance_scale,
            ).images
        dt = time.time() - t0

        # images[1] is the stylized content image
        result = images[1]
        # Resize output to match dataset size if needed
        if img_size != 512:
            result = result.resize((img_size, img_size), Image.LANCZOS)
        result.save(str(out_path))
        timing_log.append((out_name, dt))
        count += 1

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start_all
            eta = elapsed / idx * (total - idx) if idx else 0
            print(f"  [{dataset_label}] {idx}/{total}  dt={dt:.2f}s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    total_time = time.time() - start_all
    print(f"  [{dataset_label}] Done. {total_time/60:.1f} min total ({total_time/max(count,1):.2f} s/img)", flush=True)

    # Save metadata
    meta = {
        "method": "style_aligned_sd15_transfer",
        "dataset": dataset_label,
        "test_dir": str(test_dir),
        "out_dir": str(out_dir),
        "img_size": img_size,
        "style_list": style_list,
        "inversion_steps": inversion_steps,
        "guidance_scale": guidance_scale,
        "inv_guidance_scale": inv_guidance_scale,
        "total_pairs": total,
        "total_generated": count,
        "total_seconds": total_time,
        "seconds_per_image": total_time / max(count, 1),
    }
    meta_path = out_dir.parent / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("StyleAligned Remote Inference — D5 + P2A + R5", flush=True)
    print("=" * 60, flush=True)

    # ── Distinct5-WikiArt (512) ──
    print("\n[1/3] Distinct5-WikiArt", flush=True)
    d5_test = Path("I:/datasets/wikiarts20_512_test")
    if d5_test.exists():
        styles = [s for s in D5_STYLES if (d5_test / s).exists()]
        print(f"  Styles found: {styles}", flush=True)
        if styles:
            pipe, handler = load_pipeline()
            out = OUT_ROOT / "distinct5" / "images"
            run_inference(pipe, d5_test, styles, D5_STYLE_PROMPTS,
                          out, D5_SIZE, "D5")
            handler.remove()
            del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        else:
            print("  SKIP: No valid D5 style dirs found", flush=True)
    else:
        print(f"  SKIP: D5 test dir not found at {d5_test}", flush=True)

    # ── Photo2Art-256 ──
    print("\n[2/3] Photo2Art-256", flush=True)
    p2a_test = Path("I:/datasets/legacy256_overfit50/test")
    if p2a_test.exists():
        styles = [s for s in P2A_STYLES if (p2a_test / s).exists()]
        print(f"  Styles found: {styles}", flush=True)
        if styles:
            pipe, handler = load_pipeline()
            out = OUT_ROOT / "photo2art256" / "images"
            run_inference(pipe, p2a_test, styles, P2A_STYLE_PROMPTS,
                          out, P2A_SIZE, "P2A")
            handler.remove()
            del pipe; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        else:
            print("  SKIP: No valid style dirs found", flush=True)
    else:
        print(f"  SKIP: P2A test dir not found at {p2a_test}", flush=True)

    # ── Random5-WikiArt ──
    print("\n[3/3] Random5-WikiArt", flush=True)
    r5_test = Path("I:/datasets/wikiarts20_512_test")
    if r5_test.exists():
        # Find available hold-out styles
        all_dirs = sorted([d.name for d in r5_test.iterdir()
                          if d.is_dir() and not d.name.startswith('.')])
        distinct5 = set(D5_STYLES)
        styles = [s for s in R5_HOLDOUT if s in all_dirs]
        if len(styles) < 5:
            # Fallback: pick 5 styles not in Distinct5
            styles = sorted([s for s in all_dirs if s not in distinct5])[:5]
        print(f"  Styles found: {styles}", flush=True)
        if styles:
            pipe, handler = load_pipeline()
            out = OUT_ROOT / "random5" / "images"
            run_inference(pipe, r5_test, styles, R5_STYLE_PROMPTS,
                          out, R5_SIZE, "R5")
            handler.remove()
            del pipe; gc.collect(); torch.cuda.empty_cache()
        else:
            print("  SKIP: No valid style dirs found", flush=True)
    else:
        print(f"  SKIP: R5 test dir not found at {r5_test}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("StyleAligned remote inference complete!", flush=True)


if __name__ == "__main__":
    main()
