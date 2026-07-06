"""Create a 256-resolution version of the wikiarts-15 test set by resizing 512 images.

Source:  I:\datasets\wikiarts15_512_test\{style}\*.jpg
Target:  I:\datasets\wikiarts15_256_test\{style}\*.jpg  (256x256, same filenames)

Only the 15 wikiarts-15 styles are processed (distinct5 excluded).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from PIL import Image

SRC_ROOT = Path(r"I:\datasets\wikiarts15_512_test")
DST_ROOT = Path(r"I:\datasets\wikiarts15_256_test")

WIKIARTS_15_STYLES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Cubism",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Mannerism_Late_Renaissance",
    "Naive_Art_Primitivism",
    "Northern_Renaissance",
    "Pop_Art",
    "Post_Impressionism",
    "Romanticism",
    "Symbolism",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    print(f"=== wikiarts-15 256 test set creation ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  src: {SRC_ROOT}", flush=True)
    print(f"  dst: {DST_ROOT}", flush=True)
    print(f"  styles: {len(WIKIARTS_15_STYLES)}", flush=True)

    if not SRC_ROOT.exists():
        print(f"ERROR: source root does not exist: {SRC_ROOT}", file=sys.stderr)
        return 1

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.time()

    for style in WIKIARTS_15_STYLES:
        src_dir = SRC_ROOT / style
        dst_dir = DST_ROOT / style
        if not src_dir.exists():
            print(f"  WARN: source style dir missing: {src_dir}", flush=True)
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in sorted(src_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            dst_path = dst_dir / img_path.name
            if dst_path.exists() and dst_path.stat().st_size > 0:
                # Skip if already exists (resumable)
                n += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((256, 256), Image.LANCZOS)
                img.save(dst_path, "JPEG", quality=95)
                n += 1
            except Exception as e:
                print(f"  ERR processing {img_path}: {e}", flush=True)
        print(f"  [{style}] {n} images", flush=True)
        total += n

    elapsed = time.time() - t0
    print(f"\nTotal: {total} images in {elapsed:.1f}s", flush=True)
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
