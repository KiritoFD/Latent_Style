#!/usr/bin/env bash
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
CLIP_CACHE="/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "=== [STEP 1] Re-infer AdaIN + WCT with ImageNet normalization fix ==="
echo "START_INFER=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 600 "$PYTHON" /mnt/c/Users/Administrator/infer_adain_wct_256.py --method both --device cuda 2>&1

echo "END_INFER=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "AdaIN count: $(ls /mnt/i/exp_256_photo2art/adain_256/images/ 2>/dev/null | wc -l)"
echo "WCT count: $(ls /mnt/i/exp_256_photo2art/wct_256/images/ 2>/dev/null | wc -l)"

echo ""
echo "=== [STEP 2] Re-evaluate AdaIN + WCT ==="
echo "START_EVAL=$(date '+%Y-%m-%dT%H:%M:%S')"

cd "$REPO"
OUT="/mnt/i/exp_256_photo2art/eval_adain_wct_256_v2.json"
timeout 1800 "$PYTHON" /mnt/c/Users/Administrator/batch_compute_photo2art.py \
    --methods-json /mnt/c/Users/Administrator/methods_adain_wct_256.json \
    --output "$OUT" \
    --device cuda \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache "$CLIP_CACHE" \
    2>&1

echo "END_EVAL=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "=== RESULTS ==="
cat "$OUT" 2>/dev/null
