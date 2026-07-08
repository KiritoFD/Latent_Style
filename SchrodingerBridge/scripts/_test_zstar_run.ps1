# Test Z-STAR script - just do dry-run to check imports and initialization
$env:PYTHONPATH = ""
& "C:\Program Files\Python312\python.exe" -c "
import sys
sys.path.insert(0, r'.')
print('Testing Z-STAR script imports...')
try:
    # Test all imports used by _run_zstar_remote.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from einops import rearrange, repeat
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    print('All imports OK')
    print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
except Exception as e:
    print(f'Import FAILED: {e}')
    import traceback
    traceback.print_exc()
"
