"""Check if MiDaS and OpenCV are available on the remote server."""
import sys

print(f"Python: {sys.version}")

try:
    import cv2
    print(f"OpenCV (cv2): {cv2.__version__} OK")
except ImportError as e:
    print(f"OpenCV: NOT available - {e}")

try:
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    print("transformers DPT: OK")
except ImportError as e:
    print(f"transformers DPT: NOT available - {e}")

try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"torch: NOT available - {e}")

# Check if MiDaS model is cached
from pathlib import Path
import os

hf_cache = Path(r"I:\Github\Latent_Style\WEAVE\exp\eval_cache\hf\hub")
if hf_cache.exists():
    print(f"\nHF cache contents:")
    for item in sorted(hf_cache.iterdir()):
        print(f"  {item.name}")
else:
    print(f"\nHF cache not found: {hf_cache}")

# Check for MiDaS specifically
for name in ["models--Intel--dpt-large", "models--Intel--dpt-hybrid-midas"]:
    p = hf_cache / name
    print(f"  {name}: {'EXISTS' if p.exists() else 'NOT FOUND'}")

# Check results directories
results_root = Path(r"I:\Github\Latent_Style\WEAVE\results\D5-512")
if results_root.exists():
    print(f"\nD5-512 results:")
    for d in sorted(results_root.iterdir()):
        if d.is_dir():
            # Count PNG files
            pngs = list(d.glob("*.png"))
            print(f"  {d.name}: {len(pngs)} PNG files")
else:
    print(f"\nResults root not found: {results_root}")

# Check test directory
test_dir = Path(r"I:\Github\Latent_Style\WEAVE\data\test")
if test_dir.exists():
    total = 0
    for style_dir in sorted(test_dir.iterdir()):
        if style_dir.is_dir():
            imgs = [p for p in style_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
            print(f"  {style_dir.name}: {len(imgs)} images")
            total += len(imgs)
    print(f"  Total source images: {total}")
