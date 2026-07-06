#!/bin/bash
echo "=== DN03 full_eval/epoch_0003 contents ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/full_eval/epoch_0003/ 2>/dev/null

echo ""
echo "=== DN03 config (first 30 lines) ==="
head -30 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN03_adain_wct/config.json 2>/dev/null

echo ""
echo "=== Batch eval log - search for DN03 errors ==="
grep -i "DN03\|error\|fail\|traceback" /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null | tail -20

echo ""
echo "=== Current running process ==="
ps aux | grep run_evaluation | grep -v grep | head -2

echo ""
echo "=== Latest log lines ==="
tail -5 /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null

echo ""
echo "=== Current time ==="
date '+%Y-%m-%d %H:%M:%S'
