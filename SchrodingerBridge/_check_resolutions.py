"""Check actual image resolutions across datasets and model outputs."""
from __future__ import annotations
import collections
from pathlib import Path
from PIL import Image


def scan_dir(root: str, label: str, max_show: int = 10) -> None:
    d = Path(root)
    if not d.exists():
        print(f"[{label}] NOT FOUND: {root}")
        return
    sizes = collections.Counter()
    exts = collections.Counter()
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            try:
                with Image.open(p) as img:
                    sizes[img.size] += 1
                    exts[p.suffix.lower()] += 1
                    n += 1
            except Exception as e:
                print(f"  ERROR reading {p.name}: {e}")
    print(f"\n=== [{label}] ===")
    print(f"Root: {root}")
    print(f"Total images: {n}")
    print(f"Extensions: {dict(exts)}")
    print(f"Top {max_show} sizes (W, H): count")
    for size, count in sizes.most_common(max_show):
        print(f"  {size}: {count}")
    if len(sizes) > max_show:
        print(f"  ... and {len(sizes) - max_show} other sizes")


print("##### DATASET RESOLUTION CHECK #####")

# 1. Test dataset (content images)
scan_dir("I:/wikiart_distinct5_samam_512_classview/test", "TEST DATASET (content)")

# 2. SaMam output images
scan_dir("I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/images", "SaMam OUTPUT")

# 3. T11 output images
scan_dir("I:/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_long30ep/full_eval/epoch_0001/images", "T11 OUTPUT")

# 4. Latents (just count files, can't open .npy as image easily but can load shape)
print("\n\n=== [LATENT FILES] ===")
lat_dirs = [
    "I:/wikiart_distinct5_samam_512_classview/test_latents",
    "I:/wikiart_distinct5_samam_512_classview/latents",
]
import numpy as np
for lat_dir in lat_dirs:
    d = Path(lat_dir)
    if not d.exists():
        print(f"NOT FOUND: {lat_dir}")
        continue
    npy_files = list(d.rglob("*.npy"))
    print(f"\nDir: {lat_dir}")
    print(f"  Total .npy files: {len(npy_files)}")
    if npy_files:
        sample = npy_files[0]
        try:
            arr = np.load(sample)
            print(f"  Sample shape: {arr.shape}")
            print(f"  Sample dtype: {arr.dtype}")
            print(f"  Sample path: {sample.name}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
        if len(npy_files) > 1:
            sample2 = npy_files[-1]
            try:
                arr2 = np.load(sample2)
                print(f"  Last shape: {arr2.shape}")
                print(f"  Last path: {sample2.name}")
            except Exception:
                pass

print("\n\n##### DONE #####")
