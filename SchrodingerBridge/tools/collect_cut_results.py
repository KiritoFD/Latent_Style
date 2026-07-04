"""Collect CUT fake_B results into standard naming format and copy to images/cut/.

Bug in remote_master_baseline_v2.py: line 770-786 iterates top-level images/ dir
instead of images/fake_B/ subdir. This script fixes that by recursively collecting
fake_B images.

File name format in fake_B: {src_style}__{src_style}__{artist_title}.png
  - First {src_style}: prefix added by master script (testA naming: {style}__{name})
  - Second {src_style}__: original file name already contains {style}__ prefix
  - {artist_title}: the actual artwork identifier

Output format: {src_style}__{artist_title}__to__{tgt_style}.png
"""
import sys
import shutil
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
OUT_ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')
RESULTS_ROOT = OUT_ROOT / 'data'
IMAGES_CUT = OUT_ROOT / 'images' / 'cut'

IMAGES_CUT.mkdir(parents=True, exist_ok=True)

total = 0
per_style = {}

for tgt_style in STYLES:
    fake_b_dir = RESULTS_ROOT / f'cut_results_{tgt_style}' / f'cut_to_{tgt_style}' / 'test_latest' / 'images' / 'fake_B'
    if not fake_b_dir.exists():
        print(f"  [{tgt_style}] fake_B dir not found, skipping")
        per_style[tgt_style] = 0
        continue

    count = 0
    for f in sorted(fake_b_dir.glob('*.png')):
        stem = f.stem  # e.g., "Early_Renaissance__Early_Renaissance__andrea-mantegna_adoration"

        # Parse: first '__' splits src_style from rest
        if '__' in stem:
            parts = stem.split('__', 1)
            src_style = parts[0]
            rest = parts[1]  # "Early_Renaissance__andrea-mantegna_adoration"
        else:
            src_style = stem
            rest = stem

        # If rest starts with src_style + '__', strip it to get artist_title
        if rest.startswith(src_style + '__'):
            artist_title = rest[len(src_style) + 2:]
        else:
            artist_title = rest

        # Build standard output name
        out_name = f'{src_style}__{artist_title}__to__{tgt_style}.png'
        out_path = IMAGES_CUT / out_name

        if not out_path.exists():
            img = Image.open(str(f)).convert('RGB')
            img.save(str(out_path))
        count += 1

    per_style[tgt_style] = count
    total += count
    print(f"  [{tgt_style}] Collected {count} images")

# Write _DONE marker only if all 5 styles have fake_B images
all_done = all(v > 0 for v in per_style.values())
if all_done:
    import time
    done_marker = IMAGES_CUT / '_DONE'
    done_marker.write_text(f'{total} images\n{time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"\n_DONE marker written to {done_marker}")
else:
    print(f"\n_NOT all styles done yet, _DONE marker NOT written")

print(f"\nTotal collected: {total}")
print(f"Per-style: {per_style}")
