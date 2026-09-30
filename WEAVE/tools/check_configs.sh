#!/bin/bash
# Check config contents - piped to remote WSL
cd /mnt/i/Github/Latent_Style/exp/620_spatial_bridge
for cfg in 620_nswd_*/config.json; do
  echo "=== $cfg ==="
  python3 -c "
import json
with open('$cfg') as f:
    c = json.load(f)
b = c.get('bridge', {})
print('  swd_noise_sigma:', b.get('swd_noise_sigma', 'MISSING'))
print('  num_epochs:', c.get('training', {}).get('num_epochs', 'MISSING'))
print('  ablation:', c.get('ablation', {}).get('name', 'MISSING'))
"
done