#!/usr/bin/env bash
# Run batch_compute_photo2art.py for pixel256 epoch_0010 generated images.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
METHODS_JSON=$REPO/scripts/methods_ours_pixel256.json
OUTPUT=/mnt/i/exp_256_photo2art/metrics_ours_pixel256_e10.json
LOG=/mnt/i/exp_256_photo2art/_batch_compute_pixel256.log

cd "$REPO"
export OMP_NUM_THREADS=4

echo "[INFO] Batch compute metrics for pixel256 epoch_0010"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 1200 "$PYTHON" -u "$REPO/scripts/batch_compute_photo2art.py" \
    --methods-json "$METHODS_JSON" \
    --output "$OUTPUT" \
    --clip-cache /mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32 \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
echo "BATCH_RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "===RESULT==="
cat "$OUTPUT" 2>/dev/null
exit $RC
