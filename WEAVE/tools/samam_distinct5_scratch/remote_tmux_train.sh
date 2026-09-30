#!/usr/bin/env bash
# Launch training inside tmux session, fully detached from SSH
# This ensures training continues even after SSH disconnect

SESSION_NAME=samam_train_7k
TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log

# Kill existing session if any
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Clear old log
> "$LOG" 2>/dev/null || true

# Start new detached tmux session running the training script
# Redirect stdout/stderr to log file inside the tmux session
tmux new-session -d -s "$SESSION_NAME" "bash $TRAIN_SCRIPT 2>&1 | tee $LOG"

echo "=== tmux session created ==="
tmux list-sessions
echo ""
echo "=== waiting 15s for initialization ==="
sleep 15
echo ""
echo "=== tmux pane output (last 30 lines) ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -30
echo ""
echo "=== train process ==="
ps aux | grep -E "train_SaMam|python.*train" | grep -v grep | head -5
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== DONE ==="
