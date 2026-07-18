"""Verify file-name alignment between raw images and packed latent cache."""
from pathlib import Path
import torch

IMG_ROOT = Path(r"F:\wikiart_distinct5_samam_512_classview_real\train")
PACKED_ROOT = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\weave_gen\data\train\.latent_cache\packed\packed")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_FILES = [
    "00_Early_Renaissance.pt",
    "01_Impressionism.pt",
    "02_Minimalism.pt",
    "03_Rococo.pt",
    "04_Ukiyo_e.pt",
]

print("=" * 80)
print("Verifying file-name alignment: raw images <-> packed latents")
print("=" * 80)
all_ok = True
for style, packed_file in zip(STYLES, STYLE_FILES):
    img_dir = IMG_ROOT / style
    img_files = sorted([f.stem for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    packed_path = PACKED_ROOT / packed_file
    payload = torch.load(packed_path, map_location="cpu", weights_only=False)
    packed_files = [Path(f).stem for f in payload["files"]]
    # normalize: packed files may have form "Early_Renaissance/subdir__name" or just "subdir__name"
    # In packed cache, files entries typically are like "Early_Renaissance/Early_Renaissance__xxx"
    # Strip directory prefix if present
    packed_stems = []
    for pf in packed_files:
        # remove everything before the first "/" or "\"
        if "/" in pf or "\\" in pf:
            pf = pf.replace("\\", "/").split("/")[-1]
        packed_stems.append(pf)
    packed_stems = sorted(packed_stems)

    img_set = set(img_files)
    packed_set = set(packed_stems)
    only_img = img_set - packed_set
    only_packed = packed_set - img_set
    match_count = len(img_set & packed_set)
    print(f"\n[{style}]")
    print(f"  raw images:    {len(img_files)}")
    print(f"  packed latents: {len(packed_stems)}")
    print(f"  matched:        {match_count}")
    if only_img:
        print(f"  ONLY in images ({len(only_img)}): {list(only_img)[:3]}")
    if only_packed:
        print(f"  ONLY in packed ({len(only_packed)}): {list(only_packed)[:3]}")
    if only_img or only_packed:
        all_ok = False

print("\n" + "=" * 80)
print(f"ALL ALIGNED: {all_ok}")
print("=" * 80)
