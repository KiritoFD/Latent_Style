#!/usr/bin/env bash
# SaMam 256 photo2art inference on legacy256_overfit50.
# Runs inside WSL (remote). Produces 750 images under
# /mnt/i/exp_256_photo2art/samam_256/images/.
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/gen_samam_256_photo2art.py
OUT_DIR=/mnt/i/exp_256_photo2art/samam_256/images

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] SaMam 256 photo2art inference (legacy256_overfit50)"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "PYTHON=$PYTHON"
echo "SCRIPT=$SCRIPT"

"$PYTHON" -u "$SCRIPT" "$@"

RC=$?
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "=== SaMam image count ==="
ls "$OUT_DIR" 2>/dev/null | wc -l
echo "RC=$RC"
exit $RC
