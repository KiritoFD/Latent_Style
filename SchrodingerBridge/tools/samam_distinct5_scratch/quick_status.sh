#!/usr/bin/env bash
# Clean status output - avoids train.log \r flooding
RESULT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
LOG=$RESULT_DIR/train.log
CKPT_DIR=$RESULT_DIR/step_checkpoints

# Extract latest step number: tr \r to \n, grep for step pattern
# train.log format: "Epoch 0: |          | NNN/? [time, speed, losses]"
STEP=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE 'Epoch 0:.*\|[[:space:]]+[0-9]+/\?' | grep -oE '[0-9]+/\?' | grep -oE '^[0-9]+' | tail -1)
TRAIN_PID=$(pgrep -f "train_SaMam" | head -1)
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
CKPT_COUNT=$(ls "$CKPT_DIR" 2>/dev/null | grep -c "^step")
MON_PID=$(pgrep -f "remote_loop_monitor" | head -1)
KEEP_PID=$(pgrep -f "wsl_keepalive" | head -1)
TMUX=$(tmux ls 2>/dev/null | grep -c samam_train_7k)

echo "=== SaMam Status $(date +%H:%M:%S) ==="
echo "Step:        ${STEP:-?} / 7000"
echo "Train PID:   ${TRAIN_PID:-DEAD}"
echo "Monitor PID: ${MON_PID:-DEAD}"
echo "Keepalive:   ${KEEP_PID:-DEAD}"
echo "tmux:        ${TMUX:-0} session"
echo "GPU:         $GPU"
echo "Checkpoints: $CKPT_COUNT (every 250 steps, expect 28 total)"

# Progress bar
if [ -n "$STEP" ]; then
    PCT=$((STEP * 100 / 7000))
    FILLED=$((PCT / 5))
    EMPTY=$((20 - FILLED))
    BAR=$(printf '=%.0s' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null)$(printf ' .%.0s' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null)
    echo "Progress:    [$BAR] ${PCT}%"
fi
