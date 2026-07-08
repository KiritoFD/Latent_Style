"""Probe remote I: drive for Photo2Art-256 and Random5-WikiArt test directories."""
from pathlib import Path

BASES = [
    r"I:\datasets",
    r"I:\legacy256_overfit50",
    r"I:\GitHub\Latent_Style\Dataset",
    r"I:\wikiart_distinct5_samam_512_classview",
    r"I:\datasets\wikiarts20_512_test",
]

print("=== Scanning for Photo2Art-256 / Random5 test dirs ===")
for base in BASES:
    p = Path(base)
    if p.exists():
        print(f"EXISTS: {base}")
        if p.is_dir():
            for d in sorted(p.iterdir())[:40]:
                if d.is_dir():
                    name_low = d.name.lower()
                    if any(k in name_low for k in ["photo2", "legacy", "cezanne", "vangogh", "hayao", "monet"]):
                        print(f"  -> {d.name}")
            # also check for subdirs like test/
            test = p / "test"
            if test.exists() and test.is_dir():
                styles = [d.name for d in sorted(test.iterdir()) if d.is_dir()]
                print(f"  test/ styles: {styles[:10]}")
    else:
        print(f"NOT FOUND: {base}")

print("\n=== wikiarts20_512_test styles ===")
r5 = Path(r"I:\datasets\wikiarts20_512_test")
if r5.exists():
    all_styles = sorted([d.name for d in r5.iterdir() if d.is_dir()])
    print(f"All {len(all_styles)} styles: {all_styles}")
    distinct5 = {"Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"}
    other = sorted([s for s in all_styles if s not in distinct5])
    print(f"Non-Distinct5: {other}")
