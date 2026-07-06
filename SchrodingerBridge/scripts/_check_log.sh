#!/bin/bash
# Check batch eval log for timing
echo "=== Last 20 lines of batch eval log ==="
tail -20 /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null || echo "Log not found"

echo ""
echo "=== Completed experiments with timing ==="
grep -E "Completed|DONE|FINISHED|✓|✗|Skip" /mnt/i/exp_256_photo2art/_ablation_batch_eval.log 2>/dev/null | tail -30

echo ""
echo "=== Current DN01 progress ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DN01_adain_off/full_eval/epoch_0003/ 2>/dev/null || echo "No output yet"

echo ""
echo "=== Check batch eval script ==="
head -50 /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/ablation_batch_eval.sh 2>/dev/null
