#!/bin/bash
echo "=== Last 10 lines of batch eval log ==="
tail -10 /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null

echo ""
echo "=== DN05 output dir ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN05_patch64/full_eval/epoch_0003/ 2>/dev/null || echo "No output dir"

echo ""
echo "=== Current time ==="
date '+%Y-%m-%d %H:%M:%S'

echo ""
echo "=== Recent completions ==="
for d in /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN*/full_eval/epoch_0003/summary.json; do
    if [ -f "$d" ]; then
        exp_name=$(echo "$d" | cut -d'/' -f7)
        mtime=$(stat -c '%y' "$d" 2>/dev/null | cut -d'.' -f1)
        echo "$mtime | $exp_name"
    fi
done | sort
