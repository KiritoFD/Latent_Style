"""Test Windows path access for compute_dino_metrics.py."""
import os
from pathlib import Path

paths = [
    r"I:\Github\Latent_Style\SchrodingerBridge\src\utils\compute_dino_metrics.py",
    r"I:\Github\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py",
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11\epoch_0005.pt",
    r"I:\datasets\wikiart_distinct5_samam_512_classview\test",
]
for p in paths:
    print(f"{'OK' if os.path.exists(p) else 'MISS':4s} {p}")

# Check if dino model is cached
import glob
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    dino_dirs = [d.name for d in hf_cache.iterdir() if "dino" in d.name.lower()]
    print(f"\nHF cache dino dirs: {dino_dirs}")
else:
    print(f"\nHF cache not found at {hf_cache}")
    # Check alternative location
    alt = Path(r"C:\Users\Administrator\.cache\huggingface\hub")
    if alt.exists():
        dino_dirs = [d.name for d in alt.iterdir() if "dino" in d.name.lower()]
        print(f"Alt HF cache dino dirs: {dino_dirs}")
