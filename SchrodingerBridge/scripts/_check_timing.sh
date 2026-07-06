#!/bin/bash
# Check timing of recent evaluations
echo "=== Recent summary.json modification times ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
for d in exp_ablation_620/*/full_eval/epoch_0003/summary.json; do
    if [ -f "$d" ]; then
        exp_name=$(echo "$d" | cut -d'/' -f2)
        mtime=$(stat -c '%y' "$d" 2>/dev/null | cut -d'.' -f1)
        echo "$mtime | $exp_name"
    fi
done | sort | tail -15

echo ""
echo "=== Current time ==="
date '+%Y-%m-%d %H:%M:%S'

echo ""
echo "=== DN01 batch progress (latest log lines) ==="
tail -3 /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null

echo ""
echo "=== Estimate ==="
echo "DN01 started: 00:21:00"
echo "Current batch: see above"
