#!/usr/bin/env bash
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
$PYTHON -c "
import torch
data = torch.load('/mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/packed/00_cezanne.pt', map_location='cpu', weights_only=False)
print('Type:', type(data))
if isinstance(data, dict):
    for k in list(data.keys())[:5]:
        v = data[k]
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={v.shape} dtype={v.dtype}')
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            print(f'  {k}: list[{len(v)}], first shape={v[0].shape} dtype={v[0].dtype}')
        else:
            print(f'  {k}: {type(v).__name__} = {v if not isinstance(v, (list,dict)) else len(v)}')
elif isinstance(data, list):
    print(f'List of {len(data)} items')
    if len(data) > 0:
        item = data[0]
        if isinstance(item, torch.Tensor):
            print(f'  first: shape={item.shape} dtype={item.dtype}')
        elif isinstance(item, dict):
            for k in list(item.keys())[:5]:
                v = item[k]
                if isinstance(v, torch.Tensor):
                    print(f'    {k}: shape={v.shape} dtype={v.dtype}')
                else:
                    print(f'    {k}: {type(v).__name__}')
"
echo "===LATENT256 CACHE FOR COMPARISON==="
$PYTHON -c "
import torch
data = torch.load('/mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/00_cezanne.pt', map_location='cpu', weights_only=False) if __import__('os').path.exists('/mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/00_cezanne.pt') else None
if data is None:
    print('latent256 cache not found, trying legacy256_overfit50/train')
    import os
    p = '/mnt/i/legacy256_overfit50/train/.latent_cache/packed/packed/00_cezanne.pt'
    if os.path.exists(p):
        data = torch.load(p, map_location='cpu', weights_only=False)
if data is not None:
    print('Type:', type(data))
    if isinstance(data, dict):
        for k in list(data.keys())[:5]:
            v = data[k]
            if isinstance(v, torch.Tensor):
                print(f'  {k}: shape={v.shape} dtype={v.dtype}')
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                print(f'  {k}: list[{len(v)}], first shape={v[0].shape} dtype={v[0].dtype}')
            else:
                print(f'  {k}: {type(v).__name__} = {v if not isinstance(v, (list,dict)) else len(v)}')
"
