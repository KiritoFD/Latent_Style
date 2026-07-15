"""Inspect p2a_256 latent_wct image naming."""
import os
from collections import Counter

d = r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\p2a_256\images"
files = sorted([f for f in os.listdir(d) if f.endswith(".png")])
print(f"Total: {len(files)}")

# Get unique prefixes (first part before __)
prefixes = Counter()
for f in files:
    parts = f.split("__")
    prefixes[parts[0]] += 1
print("\nPrefixes:")
for p, c in sorted(prefixes.items()):
    print(f"  {p}: {c}")

# Show first 5 files
print("\nFirst 5 files:")
for f in files[:5]:
    print(f"  {f}")

# Check if any contain p2a domain names
p2a_domains = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
for dom in p2a_domains:
    matches = [f for f in files if dom in f]
    print(f"\n  '{dom}' in filename: {len(matches)} matches")
    if matches:
        print(f"    example: {matches[0]}")
