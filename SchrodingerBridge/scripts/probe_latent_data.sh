#!/bin/bash
echo "=== packed dir ==="
ls /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/
echo "=== packed/packed ==="
ls /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/packed/ 2>&1
echo "=== manifest ==="
cat /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/manifest.json 2>&1 | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps({k:v if not isinstance(v,list) else f'[{len(v)} items]' for k,v in d.items()}, indent=2))" 2>&1
echo "=== sample latent shape ==="
/home/xy/venvs/samam312/bin/python -c "
import torch
import os
d = '/mnt/i/wikiart_distinct5_samam_512_latent256/train/Early_Renaissance'
files = sorted([f for f in os.listdir(d) if f.endswith('.pt')])
if files:
    t = torch.load(os.path.join(d, files[0]), map_location='cpu', weights_only=False)
    if isinstance(t, dict):
        print('keys:', list(t.keys()))
        if 'latent' in t:
            t = t['latent']
    print('shape:', t.shape, 'dtype:', t.dtype)
    print('range:', t.min().item(), t.max().item())
"
