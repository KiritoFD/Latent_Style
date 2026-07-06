#!/usr/bin/env bash
set -uo pipefail
echo "===ABLATION STATUS REPORT==="
echo "exp_ablation_620 (trained):"
for d in /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/*/; do
    name=$(basename "$d")
    ckpt_count=$(find "$d" -name "epoch_*.pt" 2>/dev/null | wc -l)
    has_summary=$(find "$d" -name "summary.json" 2>/dev/null | wc -l)
    has_full_eval=$(find "$d" -type d -name "full_eval" 2>/dev/null | wc -l)
    last_ckpt=$(find "$d" -name "epoch_*.pt" 2>/dev/null | sort | tail -1 | xargs basename 2>/dev/null)
    echo "  $name: ckpts=$ckpt_count summary=$has_summary full_eval=$has_full_eval last=$last_ckpt"
done

echo ""
echo "===infra_I0_baseline (the baseline)==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/infra_I0_baseline/ 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/infra_I0_baseline -name "*.json" -o -name "*.pt" 2>/dev/null | head -10

echo ""
echo "===SAMPLE summary.json from DA01==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1 -name "summary.json" 2>/dev/null | head -3
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/full_eval/ 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1 -name "*.json" 2>/dev/null | head -10
