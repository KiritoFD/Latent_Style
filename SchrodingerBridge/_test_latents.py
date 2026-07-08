"""Check latent file integrity."""
import torch
import glob
import os
import sys

cache_dir = r"I:\wikiart_distinct5_samam_512_latents_ema\train"
pattern = os.path.join(cache_dir, "**", "*.pt")
files = sorted(glob.glob(pattern, recursive=True))
print(f"Found {len(files)} .pt files")

bad = 0
ok = 0
for i, f in enumerate(files[:200]):
    try:
        obj = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            keys = list(obj.keys())
            if "latent" not in obj:
                print(f"BAD (dict without 'latent' key): {f}  keys={keys}")
                bad += 1
            else:
                ok += 1
        elif isinstance(obj, torch.Tensor):
            ok += 1
        else:
            print(f"BAD (type={type(obj).__name__}): {f}")
            bad += 1
    except Exception as e:
        print(f"ERROR loading {f}: {e}")
        bad += 1

print(f"\nChecked first 200 files: {ok} OK, {bad} BAD")
