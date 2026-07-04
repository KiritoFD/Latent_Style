"""Check image sizes for SaMam and T11 output images."""
from pathlib import Path
from PIL import Image
import numpy as np

samam_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\images")
t11_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images")

import re
def _normalize_name(name):
    base = re.sub(r"\.png$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"_+", "_", base)
    return base

t11_map = {}
for p in t11_dir.glob("*.png"):
    t11_map[_normalize_name(p.name)] = p

samam_files = sorted(samam_dir.glob("*.png"))
print(f"SaMam files: {len(samam_files)}, T11 mapped: {len(t11_map)}")

# Check first 5 matched pairs
checked = 0
for sp in samam_files:
    norm = _normalize_name(sp.name)
    tp = t11_map.get(norm)
    if tp is None:
        continue
    s_img = Image.open(sp)
    t_img = Image.open(tp)
    print(f"  {sp.name[:60]}...")
    print(f"    SaMam: {s_img.size} ({s_img.mode})")
    print(f"    T11:   {t_img.size} ({t_img.mode})")
    checked += 1
    if checked >= 5:
        break

# Check size distribution
s_sizes = {}
t_sizes = {}
for sp in samam_files[:50]:
    s_sizes[Image.open(sp).size] = s_sizes.get(Image.open(sp).size, 0) + 1
for tp in list(t11_map.values())[:50]:
    t_sizes[Image.open(tp).size] = t_sizes.get(Image.open(tp).size, 0) + 1
print(f"\nSaMam size distribution (first 50): {s_sizes}")
print(f"T11 size distribution (first 50): {t_sizes}")
