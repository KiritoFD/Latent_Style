"""Generate SaMam baseline images for full WikiArt-20 (20 styles x 20 sources x 30 = 12000 pairs).

Memory-optimized v3:
- fp16 (half precision)
- empty_cache + gc.collect() after EVERY image
- batch_size=1
- 20 styles (exemplar-based, uses style reference image)

SaMam is exemplar-based: stylize(content, style_ref) -> output
So it can support any number of styles by using reference images from each style.
"""
import os
import sys
import time
import gc
import random
from pathlib import Path

SAMAM_REPO = r"I:\Github\Latent_Style\Related_Works\repos\SaMam"
sys.path.insert(0, SAMAM_REPO)

import torch
from PIL import Image
from torchvision.utils import save_image

from TEST import test_utils
from TRAIN.lightning_module.lightningmodel import LightningModel

TEST_DIR = Path(r"I:\datasets\wikiarts20_512_test")
OUTPUT_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20")
CKPT = os.path.join(SAMAM_REPO, "TRAIN", "final_model.ckpt")

WIKI20_STYLES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Cubism", "Early_Renaissance", "Expressionism", "Fauvism",
    "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism",
    "Naive_Art_Primitivism", "Northern_Renaissance", "Pop_Art", "Post_Impressionism",
    "Rococo", "Romanticism", "Symbolism", "Ukiyo_e",
]

IMAGE_SIZE = 512
STYLE_SIZE = 512
MAX_SRC_PER_STYLE = 30
SEED = 42
PROGRESS_EVERY = 25

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
    # Use fp16 for inference to save VRAM
    model = model.to(device).eval()
    model.half()
    return model


def stylize(model, content_path, style_path, device, style_size=STYLE_SIZE):
    content_img = test_utils.load(content_path)
    style_img = test_utils.load(style_path)

    content_t = test_utils.content_transforms()(content_img)
    style_t = test_utils.style_transforms(style_size)(style_img)

    content_t = content_t.to(device).unsqueeze(0).half()
    style_t = style_t.to(device).unsqueeze(0).half()

    with torch.no_grad():
        output = model.forward(content_t, style_t)
    out = output[0].detach().cpu().float()

    del content_t, style_t, output
    return out


def main():
    print(f"=== SaMam WikiArt-20 full (v3 fp16) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  ckpt: {CKPT}", flush=True)
    print(f"  styles: {len(WIKI20_STYLES)}", flush=True)

    rng = random.Random(SEED)

    # Collect source images from all 20 styles
    src_images = {}
    for style in WIKI20_STYLES:
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
        print(f"  {style}: {len(images)} sources", flush=True)

    total_src = sum(len(v) for v in src_images.values())
    total = total_src * len(WIKI20_STYLES)
    print(f"  {total_src} srcs x {len(WIKI20_STYLES)} styles = {total} images", flush=True)

    # Collect style reference images (first image from each style)
    style_refs = {}
    for style in WIKI20_STYLES:
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
    print(f"  free VRAM before model: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    model = load_model(CKPT, device)
    print(f"  free VRAM after model (fp16): {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    n_new = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    last_progress = 0

    for src_style, files in src_images.items():
        for src_path in files:
            src_stem = src_path.stem
            for tgt_style in WIKI20_STYLES:
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    n_skip += 1
                    continue

                style_ref_path = style_refs.get(tgt_style)
                if style_ref_path is None:
                    continue

                try:
                    output = stylize(model, src_path, style_ref_path, device, STYLE_SIZE)
                    save_image(output.clamp(0, 1), str(out_path))
                    n_new += 1
                except Exception as e:
                    print(f"\n  ERROR: {src_style}->{tgt_style}: {e}", flush=True)
                    n_fail += 1

                # Aggressive memory cleanup EVERY image
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
                          f"fail={n_fail}  rate={rate:.2f}/s  eta={eta/3600:.1f}h  VRAM={vram:.2f}GB", flush=True)

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
