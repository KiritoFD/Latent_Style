"""Fix CUT image naming: add double style prefix to match src_lookup.

Current:  {src_style}__{artist_title}__to__{tgt_style}.png
Target:   {src_style}__{src_style}__{artist_title}__to__{tgt_style}.png

This matches the Identity naming format and src_lookup keys.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
EVAL_CUT_IMAGES = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut\images')

if not EVAL_CUT_IMAGES.exists():
    print(f"ERROR: {EVAL_CUT_IMAGES} not found")
    sys.exit(1)

style_set = set(STYLES)
renamed = 0
skipped = 0
already_correct = 0

for f in sorted(EVAL_CUT_IMAGES.glob('*.png')):
    stem = f.stem
    if '__to__' not in stem:
        print(f"  SKIP (no __to__): {f.name}")
        skipped += 1
        continue

    left, tgt_style = stem.rsplit('__to__', 1)
    if tgt_style not in style_set:
        print(f"  SKIP (bad tgt): {f.name}")
        skipped += 1
        continue

    if '__' not in left:
        print(f"  SKIP (no __ in left): {f.name}")
        skipped += 1
        continue

    src_style, rest = left.split('__', 1)

    if src_style not in style_set:
        print(f"  SKIP (bad src): {f.name}")
        skipped += 1
        continue

    # Check if already has double prefix: src_style == rest prefix
    if rest.startswith(src_style + '__'):
        already_correct += 1
        continue

    # Build new name with double prefix
    new_stem = f'{src_style}__{src_style}__{rest}__to__{tgt_style}'
    new_name = f'{new_stem}.png'
    new_path = f.parent / new_name

    if new_path.exists():
        # Target already exists, just remove the old one
        f.unlink()
        renamed += 1
        continue

    f.rename(new_path)
    renamed += 1

print(f"\nRenamed: {renamed}")
print(f"Already correct: {already_correct}")
print(f"Skipped: {skipped}")

# Verify
total = len(list(EVAL_CUT_IMAGES.glob('*.png')))
print(f"Total PNG files now: {total}")

# Quick verification: check first file matches src_lookup format
files = sorted(EVAL_CUT_IMAGES.glob('*.png'))
if files:
    print(f"\nFirst file: {files[0].name}")
    stem = files[0].stem
    left, tgt = stem.rsplit('__to__', 1)
    src_style, rest = left.split('__', 1)
    print(f"  src_style={src_style!r}, rest={rest[:50]!r}..., tgt={tgt!r}")

print("\n==FIX_DONE==")
