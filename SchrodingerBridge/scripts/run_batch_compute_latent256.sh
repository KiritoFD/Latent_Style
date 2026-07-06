#!/usr/bin/env bash
# Run batch_compute_photo2art.py on Ours latent256 epoch_0010 images.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
METHODS_JSON=/mnt/c/Users/Administrator/methods_ours_latent256.json
OUTPUT=/mnt/i/exp_256_photo2art/eval_ours_latent256_e10.json
LOG=/mnt/i/exp_256_photo2art/_batch_compute_latent256.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Batch compute metrics for Ours latent256 epoch_0010"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 600 "$PYTHON" -u "$REPO/scripts/batch_compute_photo2art.py" \
    --methods-json "$METHODS_JSON" \
    --output "$OUTPUT" \
    --clip-cache /mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32 \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
echo "BATCH_RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"

echo "===RESULTS==="
cat "$OUTPUT" 2>/dev/null | python3 -m json.tool 2>/dev/null || cat "$OUTPUT" 2>/dev/null
exit $RC
