#!/usr/bin/env bash
# Run pixel256 training in FOREGROUND (known to work) - no background, no setsid, no tmux.
# This script will be executed via SSH and the SSH session stays open.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CONFIG=$REPO/configs/630_pixel_256_photo2art.json
LOG=/mnt/i/exp_256_photo2art/_train_pixel256_photo2art.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Train pixel256 photo2art (foreground, no eval per epoch)"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

# Run training directly, output to log file AND stdout
# pixel256 uses batch_size=1 + softmax attention, ~1.6s/step at 256x256 spatial
# 5 epochs × 10361 steps × ~1.6s = ~23h, timeout 86400s (24h) for safety
timeout 86400 "$PYTHON" -u "$REPO/run.py" --config "$CONFIG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "TRAIN_RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
exit $RC
