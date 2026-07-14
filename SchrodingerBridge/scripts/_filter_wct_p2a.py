"""Filter latent_wct p2a_256 images to 750 pairs using wikiart parsing."""
import os
import shutil
from pathlib import Path
from collections import defaultdict

base = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline")

# P2A: images use wikiart naming from wikiarts15_256_test
# Pick 5 styles that exist in both wikiarts15 and match r5 styles for consistency
# Actually, pick the 5 d5 styles if they exist, else pick first 5
p2a_img_dir = base / "p2a_256" / "images"
p2a_filtered = base / "p2a_256" / "images_750"
p2a_filtered.mkdir(exist_ok=True)

# Get all unique source styles from image filenames
# Format: {SrcStyle}__{SrcStyle}__{artist}__to__{TgtStyle}.png
src_styles = set()
tgt_styles = set()
for f in p2a_img_dir.iterdir():
    if f.suffix.lower() != ".png":
        continue
    stem = f.stem
    if "__to__" not in stem:
        continue
    left, tgt = stem.rsplit("__to__", 1)
    tgt_styles.add(tgt)
    # Source style is the first part before __
    parts = left.split("__")
    if parts:
        src_styles.add(parts[0])

print(f"P2A source styles: {sorted(src_styles)}")
print(f"P2A target styles: {sorted(tgt_styles)}")

# Pick 5 styles: use the first 5 sorted
selected = sorted(src_styles)[:5]
print(f"Selected 5 styles: {selected}")

# Filter: keep 30 per src-tgt combo
counts = defaultdict(int)
max_per_pair = 30
kept = 0
for f in sorted(p2a_img_dir.iterdir()):
    if f.suffix.lower() != ".png":
        continue
    stem = f.stem
    if "__to__" not in stem:
        continue
    left, tgt = stem.rsplit("__to__", 1)
    parts = left.split("__")
    src_style = parts[0] if parts else None
    if src_style not in selected or tgt not in selected:
        continue
    key = (src_style, tgt)
    if counts[key] < max_per_pair:
        dst = p2a_filtered / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        counts[key] += 1
        kept += 1

print(f"\nP2A: kept {kept} images (target: 5*5*30=750)")
print(f"  counts: {dict(counts)}")

# Also check what test_dir to use for DINO
p2a_test = Path(r"I:\datasets\wikiarts15_256_test")
if p2a_test.is_dir():
    print(f"\nP2A test dir exists: {p2a_test}")
    print(f"  contents: {sorted([d.name for d in p2a_test.iterdir() if d.is_dir()])}")
else:
    print(f"\nP2A test dir NOT found: {p2a_test}")
