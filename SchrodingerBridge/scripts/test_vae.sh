#!/bin/bash
/home/xy/venvs/samam312/bin/python -c "
import sys
sys.path.insert(0, '/mnt/i/Github/Latent_Style/SchrodingerBridge/src')
from utils.inference import load_vae
import torch
vae = load_vae(device='cuda', model_id='ema', cache_dir='/mnt/i/Github/Latent_Style/eval_cache/hf', enable_xformers=False)
print('VAE loaded:', type(vae).__name__)
print('VRAM after VAE load:')
import subprocess
out = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
print(f'  {out} MiB')
# Test encode/decode
x = torch.randn(1, 3, 256, 256, device='cuda') * 0.5
with torch.no_grad():
    z = vae.encode(x).latent_dist.sample() * 0.18215
    print(f'Latent shape: {z.shape}')
    y = vae.decode(z / 0.18215).sample
    print(f'Decoded shape: {y.shape}')
print('VAE OK')
"
