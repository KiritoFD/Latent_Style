"""Prepare SaMam images for evaluation: copy to eval/samam/images/ and fix naming."""
import sys
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')
SAMAM_SRC = ROOT / 'images' / 'samam'
EVAL_SAMAM = ROOT / 'eval' / 'samam' / 'images'

EVAL_SAMAM.mkdir(parents=True, exist_ok=True)

if not SAMAM_SRC.exists():
    print(f"ERROR: {SAMAM_SRC} not found")
    sys.exit(1)

# Check _DONE marker
done_marker = SAMAM_SRC / '_DONE'
if done_marker.exists():
    print(f"_DONE marker: {done_marker.read_text().strip()}")

# List source files
src_files = sorted(SAMAM_SRC.glob('*.png'))
print(f"Source files: {len(src_files)}")
if src_files:
    print(f"First 3:")
    for f in src_files[:3]:
        print(f"  {f.name}")

# Copy and fix naming: add double style prefix
style_set = set(STYLES)
copied = 0
renamed = 0
already_correct = 0

for f in src_files:
    stem = f.stem
    if '__to__' not in stem:
        print(f"  SKIP (no __to__): {f.name}")
        continue

    left, tgt_style = stem.rsplit('__to__', 1)
    if tgt_style not in style_set:
        print(f"  SKIP (bad tgt): {f.name}")
        continue

    if '__' not in left:
        print(f"  SKIP (no __ in left): {f.name}")
        continue

    src_style, rest = left.split('__', 1)

    if src_style not in style_set:
        print(f"  SKIP (bad src): {f.name}")
        continue

    # Check if already has double prefix
    if rest.startswith(src_style + '__'):
        new_stem = stem  # Already correct
        already_correct += 1
    else:
        # Build new name with double prefix
        new_stem = f'{src_style}__{src_style}__{rest}__to__{tgt_style}'
        renamed += 1

    new_name = f'{new_stem}.png'
    dst = EVAL_SAMAM / new_name

    if not dst.exists():
        shutil.copy2(str(f), str(dst))
    copied += 1

print(f"\nCopied: {copied}")
print(f"Renamed (single->double prefix): {renamed}")
print(f"Already correct (double prefix): {already_correct}")

# Verify
total = len(list(EVAL_SAMAM.glob('*.png')))
print(f"Total PNG in eval/samam/images: {total}")

# Show first file
files = sorted(EVAL_SAMAM.glob('*.png'))
if files:
    print(f"\nFirst file: {files[0].name}")

print("\n==PREP_SAMAM_DONE==")
