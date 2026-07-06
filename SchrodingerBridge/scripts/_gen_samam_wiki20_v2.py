"""Generate SaMam baseline images for WikiArt-20 Distinct5 (memory-leak fixed version).

Fixed version:
- Calls torch.cuda.empty_cache() after each image
- Times each image and reports slow outliers (>30s)
- Resumes from existing files
- Progress saved every 25 images
"""
import os
import sys
import time
import random
import gc
from pathlib import Path

SAMAM_REPO = r"I:\Github\Latent_Style\Related_Works\repos\SaMam"
sys.path.insert(0, SAMAM_REPO)

import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from TEST import test_utils
from TRAIN.lightning_module.lightningmodel import LightningModel

TEST_DIR = Path(r"I:\datasets\wikiarts20_512_test")
OUTPUT_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20")
CKPT = os.path.join(SAMAM_REPO, "TRAIN", "final_model.ckpt")

STYLES = [
    "Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e",
]

IMAGE_SIZE = 512
STYLE_SIZE = 512
MAX_SRC_PER_STYLE = 30
SEED = 42
PROGRESS_EVERY = 25
SLOW_THRESHOLD = 30.0  # seconds

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_model(ckpt_path, device):
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=device,
        nVSSMs=2, nSAVSSMs=2, nSAVSSGs=2,
        embed_dim=256, patch_size=8,
        representation_dim=64, d_state=16, expand=2.0,
        compress_ratio=8, squeeze_factor=8,
        mamba_from_trion=1,
    )
    return model.to(device).eval()


def stylize(model, content_path, style_path, device, style_size=STYLE_SIZE):
    content_img = test_utils.load(content_path)
    style_img = test_utils.load(style_path)

    content_t = test_utils.content_transforms()(content_img)
    style_t = test_utils.style_transforms(style_size)(style_img)

    content_t = content_t.to(device).unsqueeze(0)
    style_t = style_t.to(device).unsqueeze(0)

    with torch.no_grad():
        output = model.forward(content_t, style_t)
    out = output[0].detach().cpu()

    del content_t, style_t, output
    return out


def main():
    print(f"=== SaMam WikiArt-20 Distinct5 (v2 fixed) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  ckpt: {CKPT}", flush=True)
    print(f"  test_dir: {TEST_DIR}", flush=True)
    print(f"  styles({len(STYLES)}): {STYLES}", flush=True)

    rng = random.Random(SEED)

    src_images = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"WARNING: {style_dir} not found", flush=True)
            continue
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(images)
        if MAX_SRC_PER_STYLE > 0:
            images = images[:MAX_SRC_PER_STYLE]
        src_images[style] = images
        print(f"  {style}: {len(images)} source images", flush=True)

    total_src = sum(len(v) for v in src_images.values())
    total = total_src * len(STYLES)
    print(f"  {total_src} srcs x {len(STYLES)} styles = {total} images", flush=True)

    style_refs = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        images = sorted(p for p in style_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images:
            style_refs[style] = images[0]

    out_dir = OUTPUT_ROOT / "samam" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)
    if existing >= total:
        print("  All images exist, skipping.", flush=True)
        (OUTPUT_ROOT / "samam" / "_DONE").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)
    print(f"  torch={torch.__version__}, cuda={torch.version.cuda}", flush=True)
    print(f"  free VRAM before model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    model = load_model(CKPT, device)
    print(f"  free VRAM after model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    n_new = 0
    n_skip = 0
    n_slow = 0
    t0 = time.time()
    last_progress_save = 0

    pbar = tqdm(total=total, desc="samam_w20_v2", initial=existing)
    for src_style, files in src_images.items():
        for src_path in files:
            src_stem = src_path.stem
            for tgt_style in STYLES:
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    n_skip += 1
                    pbar.update(1)
                    continue

                style_ref_path = style_refs.get(tgt_style)
                if style_ref_path is None:
                    pbar.update(1)
                    continue

                t_img = time.time()
                try:
                    output = stylize(model, src_path, style_ref_path, device, STYLE_SIZE)
                    save_image(output.clamp(0, 1), str(out_path))
                    n_new += 1
                except Exception as e:
                    print(f"\n  ERROR: {src_style}->{tgt_style}: {e}", flush=True)
                dt = time.time() - t_img

                if dt > SLOW_THRESHOLD:
                    n_slow += 1
                    print(f"\n  SLOW: {src_style}->{tgt_style} took {dt:.1f}s", flush=True)
                    # Force cleanup on slow images
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"  VRAM after cleanup: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)
                else:
                    # Periodic empty_cache every 5 images
                    if (n_new + n_skip) % 5 == 0:
                        torch.cuda.empty_cache()

                pbar.update(1)

                # Progress save
                if (n_new + n_skip) - last_progress_save >= PROGRESS_EVERY:
                    last_progress_save = n_new + n_skip
                    elapsed = time.time() - t0
                    rate = (n_new + n_skip - existing) / max(elapsed, 1)
                    eta = (total - n_new - n_skip) / max(rate, 0.01)
                    print(f"  progress: {n_new + n_skip}/{total}  new={n_new} skip={n_skip} slow={n_slow}  rate={rate:.2f}/s  eta={eta/60:.1f}min  VRAM={torch.cuda.mem_get_info()[0]/1e9:.2f}GB", flush=True)

    pbar.close()
    elapsed = time.time() - t0
    print(f"  DONE: {n_new} new + {n_skip} skipped in {elapsed:.1f}s ({n_slow} slow)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    (OUTPUT_ROOT / "samam" / "_DONE").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
