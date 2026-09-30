#!/usr/bin/env bash
echo "===== SeeDream existing summary.json ====="
SEEDREAM=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750
python3 -c "
import json
d = json.load(open('$SEEDREAM/summary.json'))
print('Keys:', list(d.keys())[:20])
# Print clip/lpips related
for k in ['clip_style', 'content_lpips', 'clip_s_delta_idt', 'n_pairs', 'overall', 'aggregate']:
    if k in d:
        print(f'{k}: {d[k]}')
# Check if nested
if 'overall' in d:
    print('overall keys:', list(d['overall'].keys())[:15])
" 2>/dev/null

echo ""
echo "===== SeeDream metrics.csv head ====="
head -2 "$SEEDREAM/metrics.csv" 2>/dev/null

echo ""
echo "===== run_evaluation.py args (grep argparse) ====="
grep -E "add_argument|--" /mnt/i/Github/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py 2>/dev/null | head -40

echo ""
echo "===== Check file naming convention ====="
echo "--- SeeDream sample ---"
ls "$SEEDREAM/images/" 2>/dev/null | head -2
echo "--- adain sample ---"
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/adain/images/ 2>/dev/null | head -2
