#!/usr/bin/env bash
# Persistent loop monitor: runs in WSL background via setsid, survives SSH disconnect
# Relies on vmIdleTimeout=-1 in .wslconfig to keep WSL VM alive
# Writes status to progress.log every 5 minutes

PROGRESS_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/progress.log
TRAIN_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
CKPT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints

echo "=== Loop monitor started $(date -Iseconds) ===" >> "$PROGRESS_LOG"

while true; do
    TS=$(date -Iseconds)
    LAST_LINE=$(tail -1 "$TRAIN_LOG" 2>/dev/null | tr -d '\r')
    CKPT_COUNT=$(ls "$CKPT_DIR" 2>/dev/null | wc -l)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    PROC=$(pgrep -f "train_SaMam" | head -1)
    if [ -z "$PROC" ]; then
        PROC="DEAD"
    fi
    echo "[$TS] proc=$PROC gpu=$GPU ckpt=$CKPT_COUNT | $LAST_LINE" >> "$PROGRESS_LOG"

    if [ "$PROC" = "DEAD" ]; then
        echo "[$TS] *** TRAIN DEAD - appending last 30 lines ***" >> "$PROGRESS_LOG"
        tail -30 "$TRAIN_LOG" >> "$PROGRESS_LOG" 2>/dev/null
        break
    fi

    sleep 300
done
