#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
PROGRESS=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/progress.log
echo "=== TRAIN.LOG TAIL ==="
tail -30 "$LOG" 2>&1
echo ""
echo "=== PROGRESS.LOG TAIL ==="
tail -10 "$PROGRESS" 2>&1
echo ""
echo "=== TMUX SESSIONS ==="
tmux ls 2>&1
echo ""
echo "=== TRAIN PROCESS ==="
ps aux | grep -E "train_SaMam|python.*train" | grep -v grep
echo ""
echo "=== GPU STATUS ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo ""
echo "=== CHECKPOINTS ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/ 2>&1 | head -30
echo ""
echo "=== KEEPALIVE LOG ==="
cat /tmp/wsl_keepalive.log 2>&1 | tail -5
