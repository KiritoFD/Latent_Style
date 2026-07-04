#!/bin/bash
echo "=== packed dir structure ==="
ls /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/
echo "=== packed/packed ==="
ls /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/packed/ 2>&1
echo "=== manifest keys ==="
/home/xy/venvs/samam312/bin/python -c "
import json
with open('/mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/manifest.json') as f:
    d = json.load(f)
print('Top-level keys:', list(d.keys()))
for k, v in d.items():
    if isinstance(v, dict):
        print(f'  {k}: keys={list(v.keys())[:3]}')
        if 'count' in v:
            print(f'    count={v[\"count\"]}')
        if 'packed_path' in v:
            print(f'    packed_path={v[\"packed_path\"]}')
        if 'shape' in v:
            print(f'    shape={v[\"shape\"]}')
    elif isinstance(v, list):
        print(f'  {k}: list of {len(v)} items, first: {v[0] if v else None}')
    else:
        print(f'  {k}: {v}')
"
