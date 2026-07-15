#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log

echo "=== LOG TAIL (last 50 lines) ==="
tail -50 "$LOG" 2>/dev/null
echo ""
echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo ""
echo "=== TRAIN PROCESS ==="
ps aux | grep -E "train_SaMam|python.*train" | grep -v grep
echo ""
echo "=== CHECKPOINTS SAVED ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/ 2>/dev/null | head -10
echo ""
echo "=== LOG SIZE ==="
wc -l "$LOG" 2>/dev/null
echo "=== DONE ==="
