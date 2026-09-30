#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${BASELINE_PYTHON:-/root/venvs/samam/bin/python}
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import torch
assert hasattr(torch, "cuda") and torch.cuda.is_available()
PY
  PYTHON_BIN=$("$SCRIPT_DIR/wsl_find_python_env.sh")
fi

GPU_LOCK_FILE=${GPU_LOCK_FILE:-/mnt/g/GitHub/Latent_Style/SchrodingerBridge/aaai2027/.local_gpu_eval.lock}
WAIT_LOCK_SECONDS=${WAIT_LOCK_SECONDS:-30}

DATA_ROOT=${DATA_ROOT:-/mnt/f/wikiarts_5_full_notest}
OUT_ROOT=${OUT_ROOT:-/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samst_wikiarts5_wsl_$(date +%Y%m%d_%H%M%S)}
STYLES=${STYLES:-Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e}
EPOCHS=${EPOCHS:-100}
MAX_STEPS=${MAX_STEPS:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
IMAGE_SIZE=${IMAGE_SIZE:-256}
STYLE_SIZE=${STYLE_SIZE:-512}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
MAX_TRAIN_PER_CLASS=${MAX_TRAIN_PER_CLASS:-0}
SKIP_STYLES_WITH_EPOCH_AT_LEAST=${SKIP_STYLES_WITH_EPOCH_AT_LEAST:-0}
STOP_AFTER_ONE_PENDING_STYLE=${STOP_AFTER_ONE_PENDING_STYLE:-0}

while [[ -f "$GPU_LOCK_FILE" ]]; do
  echo "[run_samst_wikiarts5_wsl] waiting for local GPU lock: $GPU_LOCK_FILE"
  sleep "$WAIT_LOCK_SECONDS"
done

cd /mnt/g/GitHub/Latent_Style
exec "$PYTHON_BIN" /mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --styles "$STYLES" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --style-size "$STYLE_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --max-train-per-class "$MAX_TRAIN_PER_CLASS" \
  --skip-styles-with-epoch-at-least "$SKIP_STYLES_WITH_EPOCH_AT_LEAST" \
  $(if [[ "$STOP_AFTER_ONE_PENDING_STYLE" == "1" ]]; then printf '%s' '--stop-after-one-pending-style'; fi)
