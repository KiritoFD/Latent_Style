#!/usr/bin/env bash
# Continuous progress logger - appends status every 5 min to progress.log
# Runs in background, independent of SSH sessions
PROGRESS_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/progress.log
TRAIN_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
CKPT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints

echo "=== Progress logger started at $(date -Iseconds) ===" >> "$PROGRESS_LOG"

while true; do
    TS=$(date -Iseconds)
    # Get last line of train log (latest progress)
    LAST_LINE=$(tail -1 "$TRAIN_LOG" 2>/dev/null)
    # Count checkpoints
    CKPT_COUNT=$(ls "$CKPT_DIR" 2>/dev/null | wc -l)
    # GPU status
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    # Process alive?
    PROC=$(ps aux | grep "train_SaMam" | grep -v grep | head -1 | awk '{print $2}')
    if [ -z "$PROC" ]; then
        PROC="DEAD"
    fi
    echo "[$TS] proc=$PROC gpu=$GPU ckpt_count=$CKPT_COUNT last_log=$LAST_LINE" >> "$PROGRESS_LOG"

    # If process is dead, exit
    if [ "$PROC" = "DEAD" ]; then
        echo "[$TS] TRAIN PROCESS DEAD - exiting logger" >> "$PROGRESS_LOG"
        # Append full last 30 lines of train log for debugging
        echo "=== Last 30 lines of train.log ===" >> "$PROGRESS_LOG"
        tail -30 "$TRAIN_LOG" >> "$PROGRESS_LOG" 2>/dev/null
        break
    fi

    sleep 300
done
