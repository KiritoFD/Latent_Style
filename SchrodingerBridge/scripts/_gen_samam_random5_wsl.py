"""SaMam Random5 generation - WSL version with real mamba_ssm kernel.

Runs in WSL using /root/samam_venv/bin/python.
Uses 512 resolution with proper triton mamba kernel (mamba_from_trion=1).
Generates 5 Distinct5 styles x 30 srcs x 5 styles = 750 pairs.
Resume-safe: skips existing PNGs.
"""
import os
import sys
import time
import gc
import random
from pathlib import Path

# WSL paths - I: drive is mounted at /mnt/i
SAMAM_REPO = "/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam"
sys.path.insert(0, SAMAM_REPO)

import torch
from PIL import Image
from torchvision.utils import save_image

from TEST import test_utils
from TRAIN.lightning_module.lightningmodel import LightningModel

TEST_DIR = Path("/mnt/i/datasets/wikiarts20_512_test")
OUTPUT_ROOT = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20")
CKPT = os.path.join(SAMAM_REPO, "TRAIN", "final_model.ckpt")

STYLES = [
    "Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e",
]

# Real mamba_ssm is efficient - use 512 (matches D5/P256 protocol)
IMAGE_SIZE = 512
STYLE_SIZE = 512
MAX_SRC_PER_STYLE = 30
SEED = 42
PROGRESS_EVERY = 10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_model(ckpt_path, device):
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=device,
        nVSSMs=2, nSAVSSMs=2, nSAVSSGs=2,
        embed_dim=256, patch_size=8,
        representation_dim=64, d_state=16, expand=2.0,
        compress_ratio=8, squeeze_factor=8,
        mamba_from_trion=1,  # Use real triton kernel (mamba_ssm installed)
    )
    return model.to(device).eval()


def stylize(model, content_path, style_path, device, style_size=STYLE_SIZE):
    content_img = test_utils.load(content_path)
    style_img = test_utils.load(style_path)

    content_t = test_utils.content_transforms()(content_img)
    style_t = test_utils.style_transforms(style_size)(style_img)

    content_t = content_t.to(device).unsqueeze(0)
    style_t = style_t.to(device).unsqueeze(0)

    output = None
    with torch.no_grad():
        output = model.forward(content_t, style_t)
    out = output[0].detach().cpu()

    del content_t, style_t, output
    return out


def main():
    print(f"=== SaMam Random5 (WSL real mamba_ssm) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  ckpt: {CKPT}", flush=True)
    print(f"  test_dir: {TEST_DIR}", flush=True)
    print(f"  styles({len(STYLES)}): {STYLES}", flush=True)
    print(f"  image_size: {IMAGE_SIZE}", flush=True)

    # Verify mamba_ssm is real (not fallback)
    try:
        import mamba_ssm
        print(f"  mamba_ssm version: {mamba_ssm.__version__}", flush=True)
    except ImportError as e:
        print(f"  FATAL: mamba_ssm not installed: {e}", flush=True)
        return 1

    rng = random.Random(SEED)

    src_images = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found", flush=True)
            continue
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(images)
        if MAX_SRC_PER_STYLE > 0:
            images = images[:MAX_SRC_PER_STYLE]
        src_images[style] = images
        print(f"  {style}: {len(images)} srcs", flush=True)

    total_src = sum(len(v) for v in src_images.values())
    total = total_src * len(STYLES)
    print(f"  total: {total_src} srcs x {len(STYLES)} styles = {total} images", flush=True)

    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            continue
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images:
            style_refs[style] = images[0]

    out_dir = OUTPUT_ROOT / "samam" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)
    if existing >= total:
        print("  All images exist, done.", flush=True)
        (OUTPUT_ROOT / "samam" / "_DONE").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)
    print(f"  torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    print(f"  free VRAM before model: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    model = load_model(CKPT, device)
    print(f"  free VRAM after model: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    # Warmup with one image
    print("  warmup...", flush=True)
    try:
        first_src = next(iter(src_images.values()))[0]
        first_ref = next(iter(style_refs.values()))
        _ = stylize(model, first_src, first_ref, device, STYLE_SIZE)
        torch.cuda.empty_cache()
        print(f"  warmup OK, VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"  warmup failed: {e}", flush=True)

    n_new = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    last_progress = 0

    for src_style, files in src_images.items():
        for src_path in files:
            src_stem = src_path.stem
            for tgt_style in STYLES:
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    n_skip += 1
                    continue

                style_ref_path = style_refs.get(tgt_style)
                if style_ref_path is None:
                    continue

                output = None
                try:
                    output = stylize(model, src_path, style_ref_path, device, STYLE_SIZE)
                    save_image(output.clamp(0, 1), str(out_path))
                    n_new += 1
                except Exception as e:
                    print(f"\n  ERROR: {src_style}->{tgt_style}: {e}", flush=True)
                    n_fail += 1
                    gc.collect()
                    torch.cuda.empty_cache()

                if output is not None:
                    del output
                gc.collect()
                torch.cuda.empty_cache()

                if (n_new + n_skip) - last_progress >= PROGRESS_EVERY:
                    last_progress = n_new + n_skip
                    elapsed = time.time() - t0
                    rate = (n_new + n_skip - existing) / max(elapsed, 1)
                    eta = (total - n_new - n_skip) / max(rate, 0.01)
                    vram = torch.cuda.mem_get_info()[0]/1e9
                    print(f"  progress: {n_new + n_skip}/{total}  new={n_new} skip={n_skip} "
                          f"fail={n_fail}  rate={rate:.2f}/s  eta={eta/60:.1f}min  VRAM={vram:.2f}GB", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {n_new} new + {n_skip} skipped + {n_fail} failed in {elapsed:.1f}s", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    (OUTPUT_ROOT / "samam" / "_DONE").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
