"""Diagnose CUT vs Identity naming format for evaluation reuse logic."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2')

dirs_to_check = [
    ('images/cut', ROOT / 'images' / 'cut'),
    ('images/identity', ROOT / 'images' / 'identity'),
    ('eval/cut', ROOT / 'eval' / 'cut'),
    ('eval/cut/images', ROOT / 'eval' / 'cut' / 'images'),
    ('eval/identity', ROOT / 'eval' / 'identity'),
    ('eval/identity/images', ROOT / 'eval' / 'identity' / 'images'),
]

print("=" * 70)
print("Diagnosing naming format for evaluation reuse")
print("=" * 70)

for name, p in dirs_to_check:
    print(f"\n[{name}] exists={p.exists()}")
    if not p.exists():
        continue
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg')])
    print(f"  Total image files: {len(files)}")
    for f in files[:3]:
        print(f"    {f.name}")
    if files:
        # Test the glob pattern used by _list_reuse_generated_files
        to_matches = list(p.glob('*_to_*.png')) + list(p.glob('*_to_*.jpg'))
        print(f"  Matching *_to_*.* pattern: {len(to_matches)}")

# Also check: does eval/cut/images exist?
eval_cut_images = ROOT / 'eval' / 'cut' / 'images'
print(f"\n[eval/cut/images] exists={eval_cut_images.exists()}")
if eval_cut_images.exists():
    files = sorted([f for f in eval_cut_images.iterdir() if f.is_file()])
    print(f"  Total files: {len(files)}")
    for f in files[:3]:
        print(f"    {f.name}")

print("\n" + "=" * 70)
print("Diagnosis complete")
