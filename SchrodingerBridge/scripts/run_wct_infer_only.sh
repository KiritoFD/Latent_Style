#!/usr/bin/env bash
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] WCT 256 inference (photo2art)"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 600 "$PYTHON" /mnt/c/Users/Administrator/infer_adain_wct_256.py --method wct --device cuda 2>&1

echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "=== WCT count ==="
ls /mnt/i/exp_256_photo2art/wct_256/images/ 2>/dev/null | wc -l
