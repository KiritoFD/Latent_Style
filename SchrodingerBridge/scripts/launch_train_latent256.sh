#!/usr/bin/env bash
# Launch latent256 training in background (nohup) so SSH disconnect doesn't kill it.
LOG=/mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
mkdir -p /mnt/i/exp_256_photo2art
nohup bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_latent256_photo2art.sh > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "LOG=$LOG"
sleep 2
echo "=== first 30 lines of log ==="
head -30 "$LOG" 2>/dev/null
