#!/bin/bash
# Check ablation progress and search for baseline 256 run_evaluation results
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620"
completed=$(ls ${EXP_ROOT}/*/full_eval/epoch_0003/summary.json 2>/dev/null | wc -l)
echo "Ablation completed: $completed / 47"
echo "Currently running:"
ps aux | grep run_evaluation | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i ~ /exp_ablation/) {print "  "$i; break}}'

echo ""
echo "=== Search for baseline 256 run_evaluation results ==="
find /mnt/i/exp_256_photo2art -name "summary.json" 2>/dev/null | head -10
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*256*" 2>/dev/null | grep -v exp_ablation | head -10

echo ""
echo "=== Check exp_256_photo2art directory ==="
ls /mnt/i/exp_256_photo2art/ 2>/dev/null | head -20

echo ""
echo "=== Check for any baseline eval directories ==="
find /mnt/i -maxdepth 3 -name "full_eval" -type d 2>/dev/null | grep -i -E "baseline|adain|wct|samst|samam|identity|seedream" | head -10
