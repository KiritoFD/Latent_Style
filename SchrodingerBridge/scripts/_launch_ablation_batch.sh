#!/usr/bin/env bash
# Launch wrapper that starts ablation_batch_eval.sh in background, detached.
LOG=/mnt/i/exp_256_photo2art/_ablation_batch_eval.log
SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/ablation_batch_eval.sh
mkdir -p /mnt/i/exp_256_photo2art
nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown
echo "LAUNCHED PID=$PID"
echo "LOG=$LOG"
sleep 3
echo "--- first 30 log lines ---"
head -30 "$LOG" 2>/dev/null
