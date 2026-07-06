#!/usr/bin/env bash
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
echo "===DA01 full config (training + full_eval sections)==="
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/config.json') as f:
    cfg = json.load(f)
print('--- training ---')
for k, v in cfg.get('training', {}).items():
    if 'eval' in k or 'cache' in k or 'test' in k or 'batch' in k:
        print(f'  {k}: {v}')
print('--- full_eval ---')
for k, v in cfg.get('full_eval', {}).items():
    print(f'  {k}: {v}')
"

echo ""
echo "===check what run_evaluation.py needs for 620_spatial_bridge==="
grep -n "620_spatial_bridge" /mnt/i/Github/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py 2>/dev/null | head -10

echo ""
echo "===sample existing full_eval runtime csv if any==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -name "full_eval_runtime.csv" 2>/dev/null | head -3
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -name "*.log" 2>/dev/null | head -5

echo ""
echo "===check if dino cache exists for eval==="
ls /mnt/i/wikiart_distinct5_samam_512_latents_ema/train/_style_cache_620/ 2>/dev/null | head -10
ls /mnt/i/eval_cache/ 2>/dev/null | head -10
ls /mnt/i/Github/Latent_Style/eval_cache/ 2>/dev/null | head -10
