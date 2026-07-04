#!/usr/bin/env bash
echo "=== TMUX SESSIONS ==="
tmux ls 2>/dev/null || echo "(no tmux)"

echo ""
echo "=== TRAIN LOG TAIL ==="
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train_resume_20k.log
if [ -f "$LOG" ]; then
    tail -30 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -30
else
    echo "(no train log yet)"
fi

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "=== PYTHON PROCS ==="
pgrep -fa train_SaMam 2>/dev/null | head -5 || echo "(no train_SaMam process)"

echo ""
echo "=== CHECKPOINT COUNT ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/step-step=*.ckpt 2>/dev/null | wc -l
