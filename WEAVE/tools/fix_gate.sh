#!/bin/bash
# Fix gate value in all smoke test configs
set -euo pipefail

EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

for cfg_dir in "$EXP_DIR"/620_nswd_*; do
    cfg="$cfg_dir/config.json"
    if [ -f "$cfg" ]; then
        echo "=== Patching $cfg ==="
        python3 -c "
import json
with open('$cfg') as f:
    c = json.load(f)
old_gate = c['model'].get('style_cross_attn_gate_init', 'NOT SET')
c['model']['style_cross_attn_gate_init'] = 0.3
with open('$cfg', 'w') as f:
    json.dump(c, f, indent=2)
print(f'  gate: {old_gate} -> 0.3')
"
    fi
done

echo ""
echo "=== Verify ==="
for cfg_dir in "$EXP_DIR"/620_nswd_*; do
    cfg="$cfg_dir/config.json"
    if [ -f "$cfg" ]; then
        python3 -c "
import json
with open('$cfg') as f:
    c = json.load(f)
print(f'{cfg_dir}: gate={c[\"model\"].get(\"style_cross_attn_gate_init\")}')
"
    fi
done