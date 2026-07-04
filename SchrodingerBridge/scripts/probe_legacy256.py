"""Probe legacy256_overfit50 dataset structure."""
from pathlib import Path
from PIL import Image

ROOT = Path("g:/GitHub/Latent_Style/Dataset/legacy256_overfit50")

print("=== Train structure ===")
train = ROOT / "train"
for d in sorted(train.iterdir()):
    if d.is_dir():
        files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        print(f"  {d.name}: {len(files)} files")

print("\n=== Test structure ===")
test = ROOT / "test"
total_test = 0
for d in sorted(test.iterdir()):
    if d.is_dir():
        files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        print(f"  {d.name}: {len(files)} files")
        total_test += len(files)

print(f"\nTotal test images: {total_test}")

# Sample image size
sample = next((test / "cezanne").glob("*.jpg"))
img = Image.open(sample)
print(f"Sample image size: {img.size}")

# Total dataset size
import os
total_size = 0
for root, dirs, files in os.walk(ROOT):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))
print(f"Total dataset size: {total_size / 1e6:.1f} MB")
