"""Scan remote I: drive for dataset paths."""
from pathlib import Path
import os

# Scan I:/datasets for all subdirs
datasets_root = Path("I:/datasets")
if datasets_root.exists():
    print("I:/datasets subdirs:")
    for d in sorted(datasets_root.iterdir()):
        if d.is_dir():
            print(f"  {d.name}")
            # Check if it has style subdirs
            subs = sorted([s.name for s in d.iterdir() if s.is_dir()])[:15]
            if subs:
                print(f"    {subs}")

# Scan I:/GitHub/Latent_Style/Dataset for datasets
dataset_alt = Path("I:/GitHub/Latent_Style/Dataset")
if dataset_alt.exists():
    print("I:/GitHub/Latent_Style/Dataset subdirs:")
    for d in sorted(dataset_alt.iterdir()):
        if d.is_dir():
            print(f"  {d.name}")

# Check wikiarts20_512_test style list fully
r5 = Path("I:/datasets/wikiarts20_512_test")
if r5.exists():
    all_styles = sorted([d.name for d in r5.iterdir() if d.is_dir()])
    print(f"wikiarts20_512_test all styles ({len(all_styles)}):")
    print(f"  {all_styles}")
    # Count images per style
    for s in all_styles[:5]:
        imgs = [f for f in (r5/s).iterdir() if f.suffix.lower() in {'.jpg','.png','.jpeg'}]
        print(f"    {s}: {len(imgs)} images")

# Check for legacy256 anywhere
for root in [Path("I:/"), Path("I:/datasets"), Path("I:/GitHub/Latent_Style")]:
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and "legacy" in d.name.lower():
                print(f"  Found legacy path: {d}")
                subs = sorted([s.name for s in d.iterdir() if s.is_dir()])[:10]
                print(f"    subdirs: {subs}")

# Check for distinct5 anywhere
for root in [Path("I:/"), Path("I:/datasets"), Path("I:/GitHub/Latent_Style")]:
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and "distinct5" in d.name.lower():
                print(f"  Found distinct5 path: {d}")
                subs = sorted([s.name for s in d.iterdir() if s.is_dir()])[:10]
                print(f"    subdirs: {subs}")
