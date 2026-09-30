"""Build the other5 test set: 5 random styles from the 22 outside distinct5.

Picks 5 styles (seed=42) from the 22 non-D5 WikiArt styles, copies 30 random
images per style from F:\\wikiart\\wikiart into Dataset/other5_512/test/{style}/.

Usage:
    python tools/build_other5_dataset.py
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

# --- Config ---
SEED = 42
NUM_STYLES = 5
NUM_IMAGES_PER_STYLE = 30

RAW_WIKIART = Path(r"F:\wikiart\wikiart")
OUTPUT_ROOT = Path(r"G:\GitHub\Latent_Style\Dataset\other5_512\test")

DISTINCT5 = {
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    # 1. List all styles in raw wikiart, exclude D5
    all_styles = sorted(
        d.name for d in RAW_WIKIART.iterdir() if d.is_dir()
    )
    outside_d5 = [s for s in all_styles if s not in DISTINCT5]
    print(f"Total styles in wikiart: {len(all_styles)}")
    print(f"Distinct5 styles excluded: {len(DISTINCT5)}")
    print(f"Styles outside D5: {len(outside_d5)}")

    # 2. Randomly pick 5 styles
    rng = random.Random(SEED)
    selected = rng.sample(outside_d5, NUM_STYLES)
    selected.sort()
    print(f"\nSelected {NUM_STYLES} styles (seed={SEED}):")
    for s in selected:
        print(f"  {s}")

    # 3. Copy 30 random images per style
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_copied = 0
    for style in selected:
        style_dir = RAW_WIKIART / style
        all_images = sorted(
            f for f in style_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        if len(all_images) < NUM_IMAGES_PER_STYLE:
            print(f"  [WARN] {style}: only {len(all_images)} images, taking all")
            chosen = all_images
        else:
            chosen = rng.sample(all_images, NUM_IMAGES_PER_STYLE)

        out_dir = OUTPUT_ROOT / style
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            dst = out_dir / img_path.name
            try:
                shutil.copy2(img_path, dst)
                total_copied += 1
            except Exception as e:
                print(f"  [ERROR] copy {img_path} -> {dst}: {e}")
        print(f"  {style}: {len(chosen)} images -> {out_dir}")

    print(f"\nTotal images copied: {total_copied}")
    print(f"Output: {OUTPUT_ROOT}")

    # 4. Save selection metadata
    meta = {
        "seed": SEED,
        "num_styles": NUM_STYLES,
        "num_images_per_style": NUM_IMAGES_PER_STYLE,
        "selected_styles": selected,
        "source": str(RAW_WIKIART),
        "output": str(OUTPUT_ROOT),
    }
    import json
    meta_path = OUTPUT_ROOT.parent / "other5_selection.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
