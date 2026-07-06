#!/usr/bin/env bash
# Launch latent256 training with setsid for full process detachment.
# setsid creates a new session, detaching from any controlling terminal.
# Combined with nohup and /dev/null redirect, this survives SSH disconnect.
set -e

LOG=/mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
PIDFILE=/mnt/i/exp_256_photo2art/_train_latent256.pid
TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_latent256_photo2art.sh

# Clean old checkpoints from debug run
echo "[INFO] Cleaning old checkpoints..."
rm -rf /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/epoch_*.pt
rm -rf /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval
echo "[INFO] Old checkpoints cleaned."

# Kill any existing training process
pkill -f "run.py.*630_latent_256_photo2art" 2>/dev/null || true
sleep 2

# Launch with setsid for full detachment
# - stdin from /dev/null (no terminal input)
# - stdout/stderr to log file
# - setsid creates new session (detaches from controlling terminal)
mkdir -p /mnt/i/exp_256_photo2art
setsid bash -c "
    exec 0</dev/null 1>'$LOG' 2>&1
    exec '$TRAIN_SCRIPT'
" &
PID=$!
echo $PID > "$PIDFILE"
echo "PID=$PID"
echo "PIDFILE=$PIDFILE"
echo "LOG=$LOG"

sleep 5
echo "===PROCESS CHECK==="
ps -p $PID -o pid,stat,comm,args 2>/dev/null || echo "PROCESS DEAD!"
echo "===LOG FIRST 10 LINES==="
head -10 "$LOG" 2>/dev/null || echo "LOG EMPTY"
