#!/bin/bash
echo "=== I7 full config ==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/625_fc_sb/from_scratch_win/init_configs/I7.json
echo ""
echo "=== from_scratch base_n1 config (data section) ==="
python3 -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/625_fc_sb/from_scratch/configs/base_n1.json') as f:
    cfg = json.load(f)
print(json.dumps(cfg.get('data', {}), indent=2))
print('---training---')
print(json.dumps(cfg.get('training', {}), indent=2))
"
