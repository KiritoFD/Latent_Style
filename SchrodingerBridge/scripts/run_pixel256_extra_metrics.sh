#!/usr/bin/env bash
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
METHODS_JSON=/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/pixel256_methods.json
OUTPUT=/mnt/i/exp_256_photo2art/eval_pixel256_extra.json
LOG=/mnt/i/exp_256_photo2art/_pixel256_extra_metrics.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

echo "[INFO] Computing MUSIQ + ART-FID for pixel256"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 1200 "$PYTHON" -u "$REPO/scripts/batch_compute_photo2art.py" \
    --methods-json "$METHODS_JSON" \
    --output "$OUTPUT" \
    --device cuda \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache /mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32 \
    2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
if [ $RC -eq 0 ] && [ -f "$OUTPUT" ]; then
    echo "[OK] pixel256 extra metrics computed"
    echo "===RESULTS==="
    cat "$OUTPUT"
else
    echo "[FAIL] pixel256 extra metrics failed"
    tail -30 "$LOG"
fi
