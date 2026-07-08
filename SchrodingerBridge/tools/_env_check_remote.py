"""Quick environment check for remote baseline setup."""
import os, sys
print("Python:", sys.version)
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import diffusers
    print("diffusers:", diffusers.__version__)
except ImportError:
    print("diffusers: NOT INSTALLED")

try:
    import einops
    print("einops: OK")
except ImportError:
    print("einops: NOT INSTALLED")

try:
    import lpips
    print("lpips: OK")
except ImportError:
    print("lpips: NOT INSTALLED")

# Check key paths
from pathlib import Path
paths = [
    Path("I:/legacy256_overfit50/test"),
    Path("I:/datasets/wikiarts20_512_test"),
    Path("I:/wikiart_distinct5_samam_512_classview/test"),
    Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp"),
    Path("I:/GitHub/Latent_Style/Related_Works/repos"),
]
for p in paths:
    exists = p.exists()
    print(f"  {p}: {'EXISTS' if exists else 'MISSING'}")
    if exists and p.is_dir():
        subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])[:10]
        print(f"    subdirs: {subdirs}")

# Check SD1.5 cache
hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
print(f"HF_HOME: {hf_home}")
import os
sd_cache = hf_home / "hub" / "models--runwayml--stable-diffusion-v1-5"
print(f"SD1.5 cached: {sd_cache.exists()}")
