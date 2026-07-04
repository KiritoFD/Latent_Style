"""Probe seedream45_api/protocol_a_800 structure."""
from pathlib import Path
from PIL import Image

ROOT = Path("g:/GitHub/Latent_Style/seedream45_api/protocol_a_800")

# List subdirs
print("=== Subdirs ===")
for d in ROOT.iterdir():
    if d.is_dir():
        files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        print(f"  {d.name}: {len(files)} images")

# Sample image sizes
print("\n=== Image sizes ===")
for d in ROOT.iterdir():
    if not d.is_dir():
        continue
    files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    if not files:
        continue
    img = Image.open(files[0])
    print(f"  {d.name}/{files[0].name}: {img.size}")

# Sample filenames from images/
print("\n=== Sample filenames from images/ ===")
img_dir = ROOT / "images"
for f in list(img_dir.glob("*.jpg"))[:5]:
    print(f"  {f.name}")

# Count by target style in images/
print("\n=== Count by target style in images/ ===")
from collections import Counter
counts = Counter()
for f in img_dir.glob("*.jpg"):
    name = f.stem
    if "_to_" in name:
        tgt = name.rsplit("_to_", 1)[1]
        counts[tgt] += 1
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
