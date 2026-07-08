"""Probe remote for Photo2Art-256 and Random5-WikiArt test directory structures."""
from pathlib import Path

# --- Photo2Art-256 ---
p2a = Path(r"I:\datasets\legacy256_overfit50")
print("=== legacy256_overfit50 ===")
if p2a.exists():
    subdirs = [d.name for d in sorted(p2a.iterdir()) if d.is_dir()]
    print(f"  Subdirs: {subdirs}")
    test = p2a / "test"
    if test.exists():
        styles = sorted([d.name for d in test.iterdir() if d.is_dir()])
        counts = {s: len(list((test/s).glob("*.jpg")) + list((test/s).glob("*.png"))) for s in styles}
        print(f"  test/ styles ({len(styles)}):")
        for s in styles:
            print(f"    {s}: {counts[s]} images")
    else:
        print("  No test/ subdir")
else:
    print("  NOT FOUND")

# --- Random5: check existing baseline eval for which 5 styles were used ---
print("\n=== Checking existing Random5 baseline data ===")
import json
r5_out = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2")
# Check if there's a random5-specific dir
for sub in ["images", "random5"]:
    p = r5_out / sub
    if p.exists():
        print(f"  Found: {p}")
        for d in sorted(p.iterdir())[:10]:
            print(f"    {d.name}")
