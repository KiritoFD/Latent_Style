#!/usr/bin/env bash
# Launch legacy256 latent pre-encode in background.
LOG=/mnt/i/exp_256_photo2art/_preencode_legacy256_latent.log
mkdir -p /mnt/i/exp_256_photo2art
nohup bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/preencode_legacy256_latent.sh > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "LOG=$LOG"
