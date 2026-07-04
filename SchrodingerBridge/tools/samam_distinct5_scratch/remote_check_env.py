import sys
import torch
print(f"python: {sys.version}")
print(f"python_exe: {sys.executable}")
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(f"gpu_mem_total_GB: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    print(f"gpu_mem_alloc_GB: {torch.cuda.memory_allocated() / 1e9:.2f}")
    print(f"gpu_mem_reserved_GB: {torch.cuda.memory_reserved() / 1e9:.2f}")

# Check key packages
try:
    import pytorch_lightning as pl
    print(f"pytorch_lightning: {pl.__version__}")
except ImportError as e:
    print(f"pytorch_lightning: MISSING - {e}")

try:
    import open_clip
    print(f"open_clip: available")
except ImportError as e:
    print(f"open_clip: MISSING - {e}")

try:
    import lpips
    print(f"lpips: available")
except ImportError as e:
    print(f"lpips: MISSING - {e}")

try:
    from lightning_fabric import LightningFabric
    print(f"lightning_fabric: available")
except ImportError as e:
    print(f"lightning_fabric: MISSING - {e}")
