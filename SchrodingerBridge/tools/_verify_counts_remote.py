"""Verify dataset image counts on remote."""
from pathlib import Path

datasets = {
    "D5": Path("I:/datasets/wikiart_distinct5_samam_512_classview/test"),
    "D5_alt": Path("I:/datasets/wikiart_distinct5_512_images/test"),
    "P2A": Path("I:/datasets/legacy256_overfit50/test"),
    "R5": Path("I:/datasets/wikiarts20_512_test"),
}

for label, root in datasets.items():
    if not root.exists():
        print(f"  {label}: MISSING at {root}")
        continue
    style_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"\n{label} at {root}: {len(style_dirs)} styles")
    for sd in style_dirs[:10]:
        imgs = [f for f in sd.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'}]
        print(f"  {sd.name}: {len(imgs)} images")
    # Check total
    total = sum(len([f for f in sd.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'}])
                for sd in style_dirs)
    print(f"  TOTAL: {total} images across {len(style_dirs)} styles")
