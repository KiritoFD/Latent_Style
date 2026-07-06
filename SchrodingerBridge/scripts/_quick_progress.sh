#!/bin/bash
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620"
completed=$(ls ${EXP_ROOT}/*/full_eval/epoch_0003/summary.json 2>/dev/null | wc -l)
echo "Ablation completed: $completed / 47"
echo "Remaining (have ckpt, no summary):"
for d in ${EXP_ROOT}/*/; do
    name=$(basename "$d")
    if [ ! -f "${d}full_eval/epoch_0003/summary.json" ] && [ -f "${d}epoch_0003.pt" ]; then
        echo "  $name"
    fi
done
echo ""
echo "Currently running:"
ps aux | grep run_evaluation | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i ~ /exp_ablation/) {print "  "$i; break}}'
