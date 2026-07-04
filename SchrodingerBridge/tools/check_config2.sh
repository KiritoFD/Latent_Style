#!/bin/bash
for exp in 620_intrinsic_v2 620_lowswd_formal 620_film_formal; do
    echo "--- $exp ---"
    python3 -c "
import json
d = json.load(open('/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/$exp/config.json'))
fe = d.get('full_eval', {})
print('save_generated_images:', fe.get('save_generated_images', 'NOT SET (default True)'))
print('full_eval keys:', list(fe.keys())[:10])
" 2>/dev/null || echo "(parse error)"
done
