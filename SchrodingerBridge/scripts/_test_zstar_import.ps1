# Test Z-STAR import and basic initialization
$env:PYTHONPATH = "C:\Users\Administrator\z-star"
& "C:\Program Files\Python312\python.exe" -c "
import sys
sys.path.insert(0, r'C:\Users\Administrator\z-star')
print('Python:', sys.version)
try:
    from zstar_pipeline import ZstarPipeline
    print('ZstarPipeline import OK')
except Exception as e:
    print(f'ZstarPipeline import FAILED: {e}')
try:
    import torch
    print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'torch import FAILED: {e}')
try:
    from diffusers import DDIMScheduler
    print('diffusers import OK')
except Exception as e:
    print(f'diffusers import FAILED: {e}')
print('DONE')
"
