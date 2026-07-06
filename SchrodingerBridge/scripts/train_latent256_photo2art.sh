#!/usr/bin/env bash
# Train latent256 on legacy256_overfit50 photo2art 5 styles.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CONFIG=$REPO/configs/630_latent_256_photo2art.json
LOG=/mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
mkdir -p /mnt/i/exp_256_photo2art

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Train latent256 photo2art"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "CONFIG=$CONFIG"
echo "LOG=$LOG"

# Run training (foreground; log to file via tee)
timeout 86400 "$PYTHON" -u "$REPO/run.py" --config "$CONFIG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "TRAIN_RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
exit $RC
