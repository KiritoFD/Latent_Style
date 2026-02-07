import glob
import json
import os
import torch

files = glob.glob('latents/*/*.pt')
print('num_files', len(files))
if files:
    x = torch.load(files[0], map_location='cpu')
    if isinstance(x, dict):
        x = x.get('latent', x)
    print('sample_file', files[0])
    print('shape', tuple(x.shape), 'dtype', str(x.dtype))

cfg = json.load(open('Thermal/src/config.json', 'r', encoding='utf-8'))
bs = cfg['training']['batch_size']
acc = cfg['training'].get('accumulation_steps', 1)
print('batch_size', bs, 'accumulation_steps', acc, 'effective_batch', bs*acc)
print('num_workers_cfg', cfg['training'].get('num_workers', None))
print('preload_data_to_gpu', cfg['training'].get('preload_data_to_gpu', False))
print('cpu_count', os.cpu_count())
