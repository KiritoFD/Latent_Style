"""WEAVE as a plug-and-play endpoint plugin for SD1.5 Img2Img.

Generates two sets of images on the D5 test set for ablation:
  - sd15_vanilla: SD1.5 Img2Img (SDEdit, empty prompt, strength=0.5)
  - sd15_weave:   SD1.5 Img2Img + WEAVE endpoint plugin (DWT + per-subband WCT)

Both modes share the SAME SD1.5 backbone (VAE + UNet). The only difference is
that sd15_weave applies the WEAVE plugin on the final latent z_0 before VAE
decode:

  z_0 (SD1.5 output)  ──>  DWT  ──>  (LL_c, LH_c, HL_c, HH_c)
  z_s (style ref)      ──>  DWT  ──>  (LL_s, LH_s, HL_s, HH_s)
  ──>  WCT-match LH/HL/HH to style, keep LL_c (content anchor)
  ──>  IDWT  ──>  z_weaved  ──>  VAE decode

This isolates the WEAVE plugin's contribution: SD1.5 handles content generation,
WEAVE injects high-frequency style texture at the endpoint in milliseconds.

Usage:
  # Full run (750 images per mode, 25 pairs):
  python tools/infer_sd_weave_plugin.py

  # Quick smoke test (1 pair, 30 images per mode):
  python tools/infer_sd_weave_plugin.py --max_pairs 1

  # Custom parameters:
  python tools/infer_sd_weave_plugin.py --num_steps 20 --strength 0.5 --adain_scale_hf 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Import WEAVE wavelet primitives + WCT helpers (standalone functions)
from wavelet import (
    dwt2_haar_multi_decompose,
    idwt2_haar_multi_reconstruct,
)
from model import _wct_match_fiber, _precompute_style_wct_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

TEST_DIR = Path(r"G:\GitHub\Latent_Style\Dataset\eval\distinct5_512\test")
OUTPUT_ROOT = _PROJECT_ROOT / "exp" / "plugin_sd15"

SD15_MODEL = "runwayml/stable-diffusion-v1-5"
NUM_IMAGES_PER_PAIR = 30
BASE_SEED = 42
IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# WEAVE plugin (standalone, extracted from WEAVE._apply_endpoint_adain)
# ---------------------------------------------------------------------------
def weave_plugin(
    z_content: torch.Tensor,
    z_style: torch.Tensor,
    adain_scale_hf: float = 0.8,
    lowpass_levels: int = 1,
) -> torch.Tensor:
    """Apply WEAVE endpoint plugin: DWT + per-subband WCT on HF, LL preserved.

    This is the core "plug-and-play" operation. It takes the SD1.5-generated
    latent z_0 and the style reference latent z_s, then:
    1. DWT-decompose both into (LL, LH, HL, HH) subbands
    2. Keep content LL (content anchor — preserves global layout)
    3. WCT-match LH/HL/HH to style (injects high-frequency brushstrokes)
    4. IDWT-reconstruct the weaved latent

    Args:
        z_content: (B, 4, H, W) content latent (SD1.5 output, VAE-scaled)
        z_style: (1, 4, H, W) style reference latent (VAE-scaled)
        adain_scale_hf: WCT blend strength for HF subbands (0=no change, 1=full WCT)
        lowpass_levels: DWT decomposition levels (1=single-level)

    Returns:
        z_weaved: (B, 4, H, W) weaved latent (same shape as z_content)
    """
    # DWT decompose content and style
    c_decomp = dwt2_haar_multi_decompose(z_content.float(), levels=lowpass_levels)
    s_decomp = dwt2_haar_multi_decompose(z_style.float(), levels=lowpass_levels)

    # LL is preserved (content anchor) — this is the key to content fidelity
    ll_K = c_decomp["ll_K"]

    # Pre-compute style WCT stats for each HF subband (for batched content)
    style_wct_stats = {}
    for k, (s_lh, s_hl, s_hh) in enumerate(s_decomp["h"]):
        # _precompute_style_wct_stats returns (s_mean, s_sqrt) or None
        st_lh = _precompute_style_wct_stats(s_lh, target_batch=z_content.shape[0])
        st_hl = _precompute_style_wct_stats(s_hl, target_batch=z_content.shape[0])
        st_hh = _precompute_style_wct_stats(s_hh, target_batch=z_content.shape[0])
        style_wct_stats[f"h{k}"] = (st_lh, st_hl, st_hh)

    # WCT-match each HF subband
    new_subs = []
    for k, (lh, hl, hh) in enumerate(c_decomp["h"]):
        s_lh, s_hl, s_hh = s_decomp["h"][k]
        st_lh, st_hl, st_hh = style_wct_stats[f"h{k}"]

        lh_new = (1.0 - adain_scale_hf) * lh + adain_scale_hf * _wct_match_fiber(
            lh, s_lh, style_stats=st_lh
        )
        hl_new = (1.0 - adain_scale_hf) * hl + adain_scale_hf * _wct_match_fiber(
            hl, s_hl, style_stats=st_hl
        )
        hh_new = (1.0 - adain_scale_hf) * hh + adain_scale_hf * _wct_match_fiber(
            hh, s_hh, style_stats=st_hh
        )
        new_subs.append((lh_new, hl_new, hh_new))

    z_weaved = idwt2_haar_multi_reconstruct(
        {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
    )
    return z_weaved.to(dtype=z_content.dtype)


# ---------------------------------------------------------------------------
# SD1.5 Img2Img with latent capture
# ---------------------------------------------------------------------------
def build_sd15_pipeline(device: str = "cuda"):
    """Load SD1.5 Img2Img pipeline in fp16 with model CPU offload."""
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD15_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def encode_image_to_latent(pipe, image: Image.Image, device: str) -> torch.Tensor:
    """Encode a PIL image to VAE latent (scaled by scaling_factor)."""
    # Preprocess: resize + normalize to [-1, 1]
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    image_tensor = pipe.image_processor.preprocess(image).to(
        device=device, dtype=pipe.dtype
    )
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent


def decode_latent_to_image(pipe, latent: torch.Tensor, device: str) -> Image.Image:
    """Decode a VAE latent (scaled) to PIL image."""
    with torch.no_grad():
        image = pipe.vae.decode(latent.to(device, pipe.dtype) / pipe.vae.config.scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if image.shape[0] == 1:
        return Image.fromarray((image[0] * 255).astype("uint8"))
    return [Image.fromarray((img * 255).astype("uint8")) for img in image]


def img2img_to_latent(
    pipe,
    init_image: Image.Image,
    strength: float = 0.5,
    num_steps: int = 20,
    generator: torch.Generator | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Run SD1.5 Img2Img denoising and return the final latent z_0 (before decode).

    Uses empty prompt (unconditional, guidance_scale=1.0) — pure SDEdit mode.
    The returned latent is VAE-scaled (multiply by scaling_factor).
    """
    # Prepare init image
    init_image = init_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    init_image_tensor = pipe.image_processor.preprocess(init_image).to(
        device=device, dtype=pipe.dtype
    )

    # Encode to latent
    with torch.no_grad():
        init_latent = pipe.vae.encode(init_image_tensor).latent_dist.sample(generator)
        init_latent = init_latent * pipe.vae.config.scaling_factor

    # Set up scheduler timesteps
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Img2Img: skip the first (1-strength) fraction of timesteps
    # strength=0.5 → start from 50% of the way through denoising
    num_warmup_steps = max(0, len(timesteps) - int(num_steps * strength))
    actual_timesteps = timesteps[num_warmup_steps:]

    if num_warmup_steps < len(timesteps):
        t_start = timesteps[num_warmup_steps]
        noise = torch.randn(
            init_latent.shape, generator=generator, device=device, dtype=pipe.dtype
        )
        latents = pipe.scheduler.add_noise(init_latent, noise, t_start)
    else:
        latents = init_latent

    # Encode empty prompt (unconditional) using tokenizer + text_encoder directly
    # (avoids deprecated _encode_prompt / encode_prompt API differences)
    with torch.no_grad():
        tokenized = pipe.tokenizer(
            [""], padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(
            tokenized.input_ids.to(device)
        )[0].to(dtype=pipe.dtype)

    # Denoising loop
    with torch.no_grad():
        for t in actual_timesteps:
            noise_pred = pipe.unet(
                latents, t, encoder_hidden_states=prompt_embeds
            ).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    return latents  # VAE-scaled z_0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_test_images(style_name: str) -> list[Path]:
    """Return sorted list of test image paths for a given style."""
    style_dir = TEST_DIR / style_name
    if not style_dir.exists():
        print(f"[WARN] Test directory not found: {style_dir}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in style_dir.iterdir() if p.suffix.lower() in exts)


def src_name_from_filename(filename: str) -> str:
    """Extract the artist_title part from '{Style}__{artist}_{title}.jpg'."""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def build_pairs():
    """Build all 5x5 source->target pairs."""
    pairs = []
    for src in STYLE_NAMES:
        for tgt in STYLE_NAMES:
            pairs.append((src, tgt))
    return pairs


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def run_generation(
    pipe,
    pairs: list,
    out_vanilla: Path,
    out_weave: Path,
    strength: float,
    num_steps: int,
    adain_scale_hf: float,
    lowpass_levels: int,
    device: str,
):
    """Generate images for both modes and record timing."""
    out_vanilla.mkdir(parents=True, exist_ok=True)
    out_weave.mkdir(parents=True, exist_ok=True)

    # Pre-encode style reference latents (one per target style)
    print("\n=== Pre-encoding style reference latents ===")
    style_latents = {}
    style_images_cache = {}
    for tgt_style in STYLE_NAMES:
        tgt_images = get_test_images(tgt_style)
        if not tgt_images:
            print(f"  [WARN] No test images for {tgt_style}, skipping")
            continue
        # Use the FIRST test image as the style reference (infer_wct.py convention)
        ref_path = tgt_images[0]
        ref_img = Image.open(ref_path).convert("RGB")
        style_images_cache[tgt_style] = ref_img
        style_latents[tgt_style] = encode_image_to_latent(pipe, ref_img, device)
        print(f"  {tgt_style}: ref={ref_path.name}, z_s shape={style_latents[tgt_style].shape}")

    # Timing records
    timing = {
        "vanilla": {"per_image": [], "total_sec": 0.0},
        "weave": {"per_image": [], "total_sec": 0.0, "weave_only_sec": []},
        "config": {
            "strength": strength,
            "num_steps": num_steps,
            "adain_scale_hf": adain_scale_hf,
            "lowpass_levels": lowpass_levels,
            "num_pairs": len(pairs),
        },
    }

    total_pairs = len(pairs)
    total_images = 0

    for pair_idx, (src_style, tgt_style) in enumerate(pairs):
        desc = f"[{pair_idx+1}/{total_pairs}] {src_style} -> {tgt_style}"
        src_images = get_test_images(src_style)
        if not src_images:
            print(f"  {desc}: SKIP (no source images)")
            continue

        src_images = src_images[:NUM_IMAGES_PER_PAIR]
        z_s = style_latents.get(tgt_style)
        if z_s is None:
            print(f"  {desc}: SKIP (no style ref for {tgt_style})")
            continue

        print(f"\n  {desc}: {len(src_images)} images")

        for img_idx, src_path in enumerate(
            tqdm(src_images, desc=desc, leave=False)
        ):
            src_name = src_name_from_filename(src_path.name)
            out_name = f"{src_style}__{src_name}__to__{tgt_style}.png"
            vanilla_path = out_vanilla / out_name
            weave_path = out_weave / out_name

            # Skip if both already exist
            if vanilla_path.exists() and weave_path.exists():
                continue

            seed = BASE_SEED + pair_idx * NUM_IMAGES_PER_PAIR + img_idx
            generator = torch.Generator(device=device).manual_seed(seed)

            try:
                src_img = Image.open(src_path).convert("RGB")

                # === Step 1: Run SD1.5 Img2Img to get z_0 (shared by both modes) ===
                t0 = time.perf_counter()
                z_0 = img2img_to_latent(
                    pipe, src_img,
                    strength=strength, num_steps=num_steps,
                    generator=generator, device=device,
                )
                t_sd = time.perf_counter() - t0

                # === Mode A: vanilla (just decode z_0) ===
                if not vanilla_path.exists():
                    t0 = time.perf_counter()
                    img_vanilla = decode_latent_to_image(pipe, z_0, device)
                    t_dec_v = time.perf_counter() - t0
                    img_vanilla.save(str(vanilla_path))
                    timing["vanilla"]["per_image"].append(t_sd + t_dec_v)
                    timing["vanilla"]["total_sec"] += t_sd + t_dec_v

                # === Mode B: WEAVE plugin (DWT + WCT on HF, then decode) ===
                if not weave_path.exists():
                    t0 = time.perf_counter()
                    z_weaved = weave_plugin(
                        z_0, z_s,
                        adain_scale_hf=adain_scale_hf,
                        lowpass_levels=lowpass_levels,
                    )
                    t_weave = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    img_weave = decode_latent_to_image(pipe, z_weaved, device)
                    t_dec_w = time.perf_counter() - t0
                    img_weave.save(str(weave_path))
                    timing["weave"]["per_image"].append(t_sd + t_weave + t_dec_w)
                    timing["weave"]["total_sec"] += t_sd + t_weave + t_dec_w
                    timing["weave"]["weave_only_sec"].append(t_weave)

                total_images += 1

                # Clear GPU cache periodically
                if img_idx % 10 == 9:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ERROR: {out_name} -> {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save timing after each pair (checkpoint)
        _save_timing(timing, out_vanilla.parent / "timing.json")

    # Final timing summary
    _save_timing(timing, out_vanilla.parent / "timing.json")
    _print_timing_summary(timing, total_images)
    return timing


def _save_timing(timing: dict, path: Path):
    """Save timing data to JSON."""
    # Compute summary stats
    summary = dict(timing)
    for mode in ["vanilla", "weave"]:
        per_img = timing[mode]["per_image"]
        if per_img:
            summary[mode]["mean_sec"] = sum(per_img) / len(per_img)
            summary[mode]["n_images"] = len(per_img)
    if timing["weave"]["weave_only_sec"]:
        w_only = timing["weave"]["weave_only_sec"]
        summary["weave"]["weave_only_mean_sec"] = sum(w_only) / len(w_only)
        summary["weave"]["weave_only_total_sec"] = sum(w_only)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _print_timing_summary(timing: dict, total_images: int):
    """Print timing comparison."""
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY ({total_images} image pairs)")
    print(f"{'='*60}")
    v = timing["vanilla"]
    w = timing["weave"]
    if v["per_image"] and w["per_image"]:
        v_mean = sum(v["per_image"]) / len(v["per_image"])
        w_mean = sum(w["per_image"]) / len(w["per_image"])
        w_only_mean = (
            sum(w["weave_only_sec"]) / len(w["weave_only_sec"])
            if w["weave_only_sec"]
            else 0
        )
        overhead_sec = w_mean - v_mean
        overhead_pct = (overhead_sec / v_mean) * 100 if v_mean > 0 else 0

        print(f"  Vanilla SD1.5:   {v_mean:.3f} sec/image  (total {v['total_sec']:.1f}s)")
        print(f"  SD1.5 + WEAVE:   {w_mean:.3f} sec/image  (total {w['total_sec']:.1f}s)")
        print(f"  WEAVE plugin only: {w_only_mean*1000:.1f} ms/image")
        print(f"  Overhead:        +{overhead_sec:.3f} sec ({overhead_pct:+.1f}%)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="WEAVE plug-and-play endpoint plugin for SD1.5 Img2Img"
    )
    parser.add_argument(
        "--num_steps", type=int, default=20,
        help="Number of denoising steps (default: 20)",
    )
    parser.add_argument(
        "--strength", type=float, default=0.5,
        help="Img2Img strength (default: 0.5)",
    )
    parser.add_argument(
        "--adain_scale_hf", type=float, default=0.8,
        help="WEAVE WCT blend strength for HF subbands (default: 0.8)",
    )
    parser.add_argument(
        "--lowpass_levels", type=int, default=1,
        help="DWT decomposition levels (default: 1)",
    )
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Limit number of style pairs (default: all 25)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["both", "vanilla", "weave"], default="both",
        help="Which mode to run (default: both)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = build_pairs()
    if args.max_pairs is not None and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    out_vanilla = OUTPUT_ROOT / "sd15_vanilla" / "images"
    out_weave = OUTPUT_ROOT / "sd15_weave" / "images"

    print(f"WEAVE Plug-and-Play Plugin Experiment")
    print(f"{'='*60}")
    print(f"  Model: {SD15_MODEL}")
    print(f"  Test set: {TEST_DIR}")
    print(f"  Pairs: {len(pairs)} ({len(pairs)*NUM_IMAGES_PER_PAIR} images per mode)")
    print(f"  Steps: {args.num_steps}, Strength: {args.strength}")
    print(f"  WEAVE: adain_scale_hf={args.adain_scale_hf}, levels={args.lowpass_levels}")
    print(f"  Output vanilla: {out_vanilla}")
    print(f"  Output weave:   {out_weave}")
    print(f"{'='*60}")

    print("\nLoading SD1.5 pipeline...")
    pipe = build_sd15_pipeline(device=args.device)
    print(f"  Pipeline loaded (dtype={pipe.dtype})")

    run_generation(
        pipe,
        pairs=pairs,
        out_vanilla=out_vanilla,
        out_weave=out_weave,
        strength=args.strength,
        num_steps=args.num_steps,
        adain_scale_hf=args.adain_scale_hf,
        lowpass_levels=args.lowpass_levels,
        device=args.device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
