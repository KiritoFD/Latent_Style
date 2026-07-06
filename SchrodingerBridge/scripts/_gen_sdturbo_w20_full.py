"""Generate SD-Turbo baseline images for full WikiArt-20 (20 styles x 20 sources x 30 = 12000 pairs).

Memory-optimized: batch_size=1, empty_cache after each image.
Skip-resumable: existing images are skipped.
"""
import os
import sys
import time
import gc
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

TEST_DIR = Path(r"I:\datasets\wikiarts20_512_test")
OUTPUT_ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20")

WIKI20_STYLES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Cubism", "Early_Renaissance", "Expressionism", "Fauvism",
    "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism",
    "Naive_Art_Primitivism", "Northern_Renaissance", "Pop_Art", "Post_Impressionism",
    "Rococo", "Romanticism", "Symbolism", "Ukiyo_e",
]

WIKI20_PROMPTS = {
    "Abstract_Expressionism": "a painting in Abstract Expressionism style, bold gestural brushstrokes",
    "Art_Nouveau_Modern": "a painting in Art Nouveau style, ornamental decorative lines",
    "Baroque": "a painting in Baroque style, dramatic chiaroscuro lighting",
    "Color_Field_Painting": "a painting in Color Field style, large flat areas of color",
    "Cubism": "a painting in Cubism style, geometric fragmented forms",
    "Early_Renaissance": "a painting in Early Renaissance style, tempera on panel",
    "Expressionism": "a painting in Expressionism style, vivid emotional colors",
    "Fauvism": "a painting in Fauvism style, wild unnatural vibrant colors",
    "High_Renaissance": "a painting in High Renaissance style, balanced classical composition",
    "Impressionism": "a painting in Impressionism style, soft brushstrokes light and color",
    "Mannerism_Late_Renaissance": "a painting in Mannerism style, elongated figures complex poses",
    "Minimalism": "a painting in Minimalism style, simple geometric forms",
    "Naive_Art_Primitivism": "a painting in Naive Art Primitivism style, simple childlike forms",
    "Northern_Renaissance": "a painting in Northern Renaissance style, detailed realistic oil technique",
    "Pop_Art": "a painting in Pop Art style, bold commercial imagery",
    "Post_Impressionism": "a painting in Post-Impressionism style, structured brushwork vivid color",
    "Rococo": "a painting in Rococo style, ornate decorative pastel colors",
    "Romanticism": "a painting in Romanticism style, dramatic emotional sublime scenery",
    "Symbolism": "a painting in Symbolism style, dreamlike metaphorical imagery",
    "Ukiyo_e": "a painting in Ukiyo-e style, Japanese woodblock print flat colors strong outlines",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = 512
MAX_SRC_PER_STYLE = 30
SEED = 42
STRENGTH = 0.5
STEPS = 4
GUIDANCE = 1.0


def main():
    print(f"=== SD-Turbo WikiArt-20 full generation ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  styles: {len(WIKI20_STYLES)}", flush=True)

    out_dir = OUTPUT_ROOT / "sdturbo" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    sources = []
    for style in WIKI20_STYLES:
        style_dir = TEST_DIR / style
        if not style_dir.exists():
            print(f"[WARN] {style_dir} not found", flush=True)
            continue
        imgs = sorted(p for p in style_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(imgs)
        if MAX_SRC_PER_STYLE > 0:
            imgs = imgs[:MAX_SRC_PER_STYLE]
        for p in imgs:
            sources.append((style, p))

    total = len(sources) * len(WIKI20_STYLES)
    print(f"  {len(sources)} srcs x {len(WIKI20_STYLES)} styles = {total} images", flush=True)

    existing = len(list(out_dir.glob("*.png")))
    print(f"  existing: {existing}/{total}", flush=True)
    if existing >= total:
        print("  All images exist, done.", flush=True)
        (OUTPUT_ROOT / "sdturbo" / "_DONE").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)
    print(f"  free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"  VRAM after model load: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB", flush=True)

    n_new = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()

    for src_style, src_path in sources:
        src_stem = src_path.stem
        try:
            content_img = Image.open(src_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        except Exception as e:
            print(f"[WARN] load fail {src_path}: {e}", flush=True)
            n_fail += 1
            continue

        for tgt_style in WIKI20_STYLES:
            out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                n_skip += 1
                continue

            try:
                prompt = WIKI20_PROMPTS[tgt_style]
                generator = torch.Generator(device=device).manual_seed(SEED)
                output = pipe(
                    prompt=prompt,
                    image=content_img,
                    strength=STRENGTH,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=generator,
                ).images[0]
                output.save(out_path)
                n_new += 1
            except Exception as e:
                print(f"[WARN] gen fail {src_style}->{tgt_style}: {e}", flush=True)
                n_fail += 1

            # Memory cleanup every image
            del output
            torch.cuda.empty_cache()

        if (n_new + n_skip) % 100 == 0:
            elapsed = time.time() - t0
            rate = (n_new + n_skip) / max(elapsed, 1)
            eta = (total - n_new - n_skip) / max(rate, 0.01)
            vram = torch.cuda.mem_get_info()[0]/1e9
            print(f"  progress: {n_new + n_skip}/{total}  new={n_new} skip={n_skip} "
                  f"fail={n_fail}  rate={rate:.1f}/s  eta={eta/3600:.1f}h  VRAM={vram:.2f}GB", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: {n_new} new + {n_skip} skipped + {n_fail} failed in {elapsed:.1f}s", flush=True)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    (OUTPUT_ROOT / "sdturbo" / "_DONE").write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
