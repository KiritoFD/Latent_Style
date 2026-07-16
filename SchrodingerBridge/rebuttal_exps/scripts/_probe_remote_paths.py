"""Probe remote paths for ArtFID-related files."""
import os
from pathlib import Path

# Check key directories
candidates = [
    "I:/Github/Latent_Style/WEAVE",
    "I:/Github/Latent_Style/WEAVE/exp",
    "I:/Github/Latent_Style/WEAVE/exp/rebuttal",
    "I:/Github/Latent_Style/WEAVE/exp/rebuttal/expA_seed7",
    "I:/Github/Latent_Style/WEAVE/experiments",
    "I:/Github/Latent_Style/WEAVE/experiments/rebuttal_20260716",
    "I:/Github/Latent_Style/WEAVE/runs/submission/robustness/early_stop_seed7",
    "I:/Github/Latent_Style/WEAVE/data/test",
]

print("=== Path existence check ===")
for p in candidates:
    exists = os.path.exists(p)
    print(f"  [{exists}] {p}")
    if exists and os.path.isdir(p):
        try:
            entries = sorted(os.listdir(p))
            if len(entries) <= 30:
                for e in entries:
                    print(f"        - {e}")
            else:
                print(f"        ({len(entries)} entries; first 20):")
                for e in entries[:20]:
                    print(f"        - {e}")
        except Exception as exc:
            print(f"        ERROR listing: {exc}")

# Find any artfid-related files
print("\n=== ArtFID-related files under I:/Github/Latent_Style/WEAVE/exp ===")
root = Path("I:/Github/Latent_Style/WEAVE/exp")
if root.exists():
    for item in root.rglob("*"):
        name = item.name.lower()
        if "artfid" in name or "art_fid" in name or "fid" in name and item.is_file():
            print(f"  {item}")
            if item.is_file() and item.suffix == ".json":
                try:
                    print(f"    size={item.stat().st_size}")
                except Exception:
                    pass

# Find baseline output directories
print("\n=== Baseline output directories ===")
baseline_candidates = [
    "I:/Github/Latent_Style/WEAVE/exp/baselines",
    "I:/Github/Latent_Style/WEAVE/baselines",
    "I:/Github/Latent_Style/WEAVE/exp/samam",
    "I:/Github/Latent_Style/WEAVE/exp/stylealigned",
    "I:/Github/Latent_Style/WEAVE/exp/zstar",
    "I:/Github/Latent_Style/WEAVE/exp/idt",
    "I:/Github/Latent_Style/WEAVE/exp/seedream",
]
for p in baseline_candidates:
    if os.path.exists(p):
        print(f"  [EXISTS] {p}")
        try:
            for e in sorted(os.listdir(p))[:15]:
                print(f"        - {e}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
    else:
        print(f"  [MISS]   {p}")

# Look for any directories with summary.json and ArtFID metric
print("\n=== Search for artfid in all summary.json under exp/ ===")
root = Path("I:/Github/Latent_Style/WEAVE/exp")
if root.exists():
    count = 0
    for item in root.rglob("summary.json"):
        try:
            text = item.read_text(encoding="utf-8", errors="ignore")
            if "artfid" in text.lower() or "art_fid" in text.lower():
                print(f"  {item}")
                count += 1
                if count >= 30:
                    print("  ... (truncated, more than 30 hits)")
                    break
        except Exception:
            pass

# Look for the ArtFID table that's currently in the paper
print("\n=== Search for *.csv with artfid ===")
root = Path("I:/Github/Latent_Style/WEAVE")
if root.exists():
    count = 0
    for item in root.rglob("*.csv"):
        name = item.name.lower()
        if "artfid" in name or "art_fid" in name:
            print(f"  {item} (size={item.stat().st_size})")
            count += 1
            if count >= 30:
                break

print("\n=== Search for *.json with artfid in filename ===")
if root.exists():
    count = 0
    for item in root.rglob("*.json"):
        name = item.name.lower()
        if "artfid" in name or "art_fid" in name:
            print(f"  {item} (size={item.stat().st_size})")
            count += 1
            if count >= 30:
                break

print("\n=== Probe complete ===")
