"""Diagnose src_lookup keys vs CUT/Identity parsed src_stem."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']

print("=" * 70)
print("Diagnosing src_lookup keys vs parsed src_stem")
print("=" * 70)

# Build src_lookup like run_evaluation.py does
src_lookup = {}
for style in STYLES:
    style_dir = TEST_DIR / style
    if not style_dir.exists():
        print(f"  WARNING: {style_dir} not found")
        continue
    for f in sorted(style_dir.iterdir()):
        if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
            src_lookup[(style, f.stem)] = str(f)

print(f"\nTotal src_lookup entries: {len(src_lookup)}")
print("\nFirst 3 entries per style:")
for style in STYLES:
    entries = [(k, v) for k, v in src_lookup.items() if k[0] == style]
    print(f"\n  [{style}] ({len(entries)} entries):")
    for k, v in entries[:3]:
        print(f"    key=({k[0]!r}, {k[1]!r})")

# Now test CUT and Identity name parsing
print("\n" + "=" * 70)
print("Testing name parsing")
print("=" * 70)

cut_samples = [
    "Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece__to__Early_Renaissance.png",
]
identity_samples = [
    "Early_Renaissance__Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece__to__Early_Renaissance.png",
]

def parse(name):
    stem = Path(name).stem
    if "__to__" in stem:
        left, tgt = stem.rsplit("__to__", 1)
        if "__" in left:
            src_style, src_stem = left.split("__", 1)
            return src_style, src_stem, tgt
    return None

for name in cut_samples:
    parsed = parse(name)
    print(f"\nCUT: {name}")
    print(f"  parsed: {parsed}")
    if parsed:
        key = (parsed[0], parsed[1])
        print(f"  src_lookup hit: {key in src_lookup}")

for name in identity_samples:
    parsed = parse(name)
    print(f"\nIdentity: {name}")
    print(f"  parsed: {parsed}")
    if parsed:
        key = (parsed[0], parsed[1])
        print(f"  src_lookup hit: {key in src_lookup}")

print("\n" + "=" * 70)
print("Diagnosis complete")
