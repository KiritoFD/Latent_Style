"""Rename Latent-WCT generated images from __to__ double-prefix format to _to_ standard format.

From: {Style}__{Style}__{artist}_{title}__to__{TgtStyle}.png
To:   {Style}_{artist}_{title}_to_{TgtStyle}.png
"""
import os
import sys
from pathlib import Path

DIR = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000\images")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

count = 0
for f in sorted(DIR.glob("*__to__*.png")):
    name = f.name
    # Strip double style prefix: {Style}__{Style}__ -> {Style}_
    for s in STYLES:
        double = f"{s}__{s}__"
        if name.startswith(double):
            name = f"{s}_" + name[len(double):]
            break
    # Replace __to__ with _to_
    name = name.replace("__to__", "_to_")
    if name != f.name:
        f.rename(f.parent / name)
        count += 1

print(f"Renamed {count} files")
# Verify
remaining = list(DIR.glob("*__to__*.png"))
print(f"Remaining __to__ files: {len(remaining)}")
new_count = len(list(DIR.glob("*_to_*.png")))
print(f"New _to_ files: {new_count}")
