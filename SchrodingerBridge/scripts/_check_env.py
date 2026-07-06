"""Check Python environment for pyiqa, diffusers, MUSIQ cache."""
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import pyiqa
    print(f"pyiqa: {pyiqa.__version__} at {pyiqa.__file__}")
except ImportError as e:
    print(f"pyiqa: NOT INSTALLED ({e})")

try:
    import diffusers
    print(f"diffusers: {diffusers.__version__}")
except ImportError as e:
    print(f"diffusers: NOT INSTALLED ({e})")

try:
    import torch
    print(f"torch: {torch.__version__}, cuda={torch.cuda.is_available()}")
except ImportError as e:
    print(f"torch: NOT INSTALLED ({e})")

# Check HF cache
import os
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hf_cache):
    models = [d for d in os.listdir(hf_cache) if d.startswith("models--")]
    print(f"HF cache models: {models}")
else:
    print(f"HF cache not found at {hf_cache}")

# Check MUSIQ weights
musiq_paths = [
    "C:\\Users\\Administrator\\musiq_koniq_ckpt-e95806b9.pth",
    "C:\\Users\\Administrator\\.cache\\torch\\hub\\pyiqa\\musiq_koniq_ckpt-e95806b9.pth",
    "C:\\Users\\Administrator\\.cache\\torch\\hub\\checkpoints\\musiq_koniq_ckpt-e95806b9.pth",
]
for p in musiq_paths:
    if os.path.exists(p):
        print(f"MUSIQ weights FOUND: {p} ({os.path.getsize(p)} bytes)")
    else:
        print(f"MUSIQ weights missing: {p}")

# Test pyiqa loading
try:
    import pyiqa
    print("\nTesting pyiqa.create_metric('musiq')...")
    m = pyiqa.create_metric("musiq", device="cpu")
    print("MUSIQ loaded OK")
except Exception as e:
    print(f"MUSIQ load FAILED: {e}")
