#!/usr/bin/env bash
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
CLIP_CACHE="/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"
OUT="/mnt/i/exp_256_photo2art/eval_samst_256.json"

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] SAMST 256 evaluation on photo2art"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 1200 "$PYTHON" /mnt/c/Users/Administrator/batch_compute_photo2art.py \
    --methods-json /mnt/c/Users/Administrator/methods_samst_only_256.json \
    --output "$OUT" \
    --device cuda \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache "$CLIP_CACHE" \
    2>&1

echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "[INFO] Results: $OUT"
cat "$OUT" 2>/dev/null
