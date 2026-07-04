"""Generate identity_256 images: copy src images as their own output for all target styles.

For each src image in legacy256_overfit50/test/{style}/, create 5 copies
(one per target style) with filename {src_style}_{id}_to_{tgt_style}.jpg.

Total: 5 styles × 30 images × 5 targets = 750 images.
"""
import shutil
from pathlib import Path

TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")
OUT_DIR = Path("/mnt/i/exp_256_photo2art/identity_256/images")
STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
for src_style in STYLES:
    src_dir = TEST_ROOT / src_style
    for src_file in sorted(src_dir.iterdir()):
        if src_file.suffix.lower() not in {".jpg", ".png"}:
            continue
        src_id = src_file.stem
        for tgt_style in STYLES:
            out_name = f"{src_style}_{src_id}_to_{tgt_style}.jpg"
            shutil.copy2(src_file, OUT_DIR / out_name)
            count += 1

print(f"Generated {count} identity images in {OUT_DIR}")
