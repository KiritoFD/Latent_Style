#!/bin/bash
echo "=== DN03 status ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/ 2>/dev/null | head -20

echo ""
echo "=== DN03 checkpoint ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/epoch_0003.pt 2>/dev/null || echo "No epoch_0003.pt"
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/checkpoint/ 2>/dev/null || echo "No checkpoint dir"

echo ""
echo "=== DN03 full_eval ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/full_eval/ 2>/dev/null || echo "No full_eval"

echo ""
echo "=== Log entries for DN03 ==="
grep -A5 "DN03" /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null | tail -20

echo ""
echo "=== Current pending count ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
pending=0
for d in exp_ablation_620/*/; do
    name=$(basename "$d")
    ckpt="$d/epoch_0003.pt"
    summary="$d/full_eval/epoch_0003/summary.json"
    if [ -f "$ckpt" ] && [ ! -f "$summary" ]; then
        pending=$((pending+1))
        echo "STILL_PENDING: $name"
    fi
done
echo "Total still pending: $pending"

echo ""
echo "=== Current time ==="
date '+%Y-%m-%d %H:%M:%S'
