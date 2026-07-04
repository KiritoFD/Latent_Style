#!/bin/bash
# Check gate value in base config
set -euo pipefail
BASE="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/config.json"
python3 -c "
import json
with open('$BASE') as f:
    cfg = json.load(f)
m = cfg.get('model', {})
print('style_cross_attn_gate_init:', m.get('style_cross_attn_gate_init', 'NOT SET'))
print('transport_prediction_mode:', m.get('transport_prediction_mode', 'NOT SET'))
print('endpoint_head_mode:', m.get('endpoint_head_mode', 'NOT SET'))
"