import sys
print("Python:", sys.version)
try:
    import pyiqa
    print("pyiqa: OK")
except ImportError as e:
    print("pyiqa: MISSING -", e)

try:
    import lpips
    print("lpips: OK")
except ImportError as e:
    print("lpips: MISSING -", e)

try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except ImportError as e:
    print("torch: MISSING -", e)

try:
    import transformers
    print("transformers:", transformers.__version__)
except ImportError as e:
    print("transformers: MISSING -", e)

import os
from pathlib import Path
# Check CLIP cache
clip_paths = list(Path(r"C:\Users\Administrator\.cache\huggingface\hub").glob("*clip*")) if Path(r"C:\Users\Administrator\.cache\huggingface\hub").exists() else []
print("CLIP cache:", clip_paths)

# Check MUSIQ weights
torch_home = r"C:\Users\Administrator\.cache\torch"
musiq_paths = list(Path(torch_home).rglob("*musiq*")) if Path(torch_home).exists() else []
print("MUSIQ weights:", musiq_paths[:3])

# Check SaMam images
samam_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images")
if samam_dir.exists():
    pngs = list(samam_dir.glob("*.png"))
    print(f"SaMam images: {len(pngs)}")
else:
    print(f"SaMam dir MISSING: {samam_dir}")

# Check test set
test_root = Path(r"I:\datasets\wikiarts20_512_test")
if test_root.exists():
    print(f"Test set: OK")
    for s in ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]:
        sd = test_root / s
        if sd.exists():
            print(f"  {s}: {len(list(sd.iterdir()))} files")
else:
    print(f"Test set MISSING: {test_root}")
