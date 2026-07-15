"""Check MD5 hash distribution of PNG outputs for AdaIN vs WCT."""
import hashlib
import os
from pathlib import Path

ADAIN_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\adain_v32k")
WCT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\wct_v32k")

for label, d in [("AdaIN", ADAIN_DIR), ("WCT", WCT_DIR)]:
    if not d.exists():
        print(f"{label}: directory not found {d}")
        continue
    files = sorted(os.listdir(d))
    hashes = []
    for f in files:
        with open(d / f, "rb") as fp:
            h = hashlib.md5(fp.read()).hexdigest()[:8]
            hashes.append(h)
    print(f"\n{label}: {len(files)} files, {len(set(hashes))} unique hashes")
    print("First 5:")
    for f, h in zip(files[:5], hashes[:5]):
        print(f"  {f[:60]}... -> {h}")

# Also check pixel value range of a few images
print("\n=== Pixel Value Range ===")
from PIL import Image
import numpy as np

for label, d in [("AdaIN", ADAIN_DIR), ("WCT", WCT_DIR)]:
    if not d.exists():
        continue
    files = sorted(os.listdir(d))[:3]
    print(f"\n{label}:")
    for f in files:
        img = Image.open(d / f).convert("RGB")
        arr = np.array(img)
        print(f"  {f[:50]}... shape={arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
