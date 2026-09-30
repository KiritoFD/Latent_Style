"""Check MD5 hash distribution of WCT VGG-19 PNG outputs."""
import hashlib
import os
from pathlib import Path
from PIL import Image
import numpy as np

WCT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\wct_vgg19")

files = sorted(os.listdir(WCT_DIR))
hashes = []
for f in files:
    with open(WCT_DIR / f, "rb") as fp:
        h = hashlib.md5(fp.read()).hexdigest()[:8]
        hashes.append(h)

print(f"WCT VGG-19: {len(files)} files, {len(set(hashes))} unique hashes")
print("First 10:")
for f, h in zip(files[:10], hashes[:10]):
    print(f"  {f[:60]}... -> {h}")

# Pixel value range
print("\n=== Pixel Value Range ===")
for f in files[:5]:
    img = Image.open(WCT_DIR / f).convert("RGB")
    arr = np.array(img)
    print(f"  {f[:50]}... shape={arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
