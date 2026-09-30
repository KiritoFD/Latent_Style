#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PYTHON_BIN=${BASELINE_PYTHON:-/root/venvs/samam/bin/python}
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import torch
assert hasattr(torch, "cuda") and torch.cuda.is_available()
PY
  PYTHON_BIN=$("$SCRIPT_DIR/wsl_find_python_env.sh")
fi

GPU_LOCK_FILE=${GPU_LOCK_FILE:-/mnt/g/GitHub/Latent_Style/SchrodingerBridge/aaai2027/.local_gpu_eval.lock}
WAIT_LOCK_SECONDS=${WAIT_LOCK_SECONDS:-30}

CONTENT_ROOT=${CONTENT_ROOT:-/mnt/f/wikiarts_5_full_notest/train_flat/content}
STYLE_ROOT=${STYLE_ROOT:-/mnt/f/wikiarts_5_full_notest/train_flat/style}
OUT_ROOT=${OUT_ROOT:-/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_wsl_$(date +%Y%m%d_%H%M%S)}
ITERATIONS=${ITERATIONS:-200000}
VAL_INTERVAL=${VAL_INTERVAL:-1000}
BATCH_SIZE=${BATCH_SIZE:-1}
TRAIN_IMAGE_SIZE=${TRAIN_IMAGE_SIZE:-256}
TRAIN_CROP_SIZE=${TRAIN_CROP_SIZE:-256}
EVAL_IMAGE_SIZE=${EVAL_IMAGE_SIZE:-256}
PATCH_SIZE=${PATCH_SIZE:-8}
PRECISION=${PRECISION:-32-true}
MAMBA_FROM_TRION=${MAMBA_FROM_TRION:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
IDENTITY_GRADIENT_CHECKPOINTING=${IDENTITY_GRADIENT_CHECKPOINTING:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-0}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-0.2}
NUM_SANITY_VAL_STEPS=${NUM_SANITY_VAL_STEPS:-0}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1}
CHECKPOINT_EVERY_N_STEPS=${CHECKPOINT_EVERY_N_STEPS:-500}

while [[ -f "$GPU_LOCK_FILE" ]]; do
  echo "[run_samam_wikiarts5_wsl] waiting for local GPU lock: $GPU_LOCK_FILE"
  sleep "$WAIT_LOCK_SECONDS"
done

cd /mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam/TRAIN
exec "$PYTHON_BIN" train_SaMam.py \
  --content "$CONTENT_ROOT" \
  --style "$STYLE_ROOT" \
  --gpus 0 \
  --iterations "$ITERATIONS" \
  --val-interval "$VAL_INTERVAL" \
  --batch-size "$BATCH_SIZE" \
  --train-image-size "$TRAIN_IMAGE_SIZE" \
  --train-crop-size "$TRAIN_CROP_SIZE" \
  --eval-image-size "$EVAL_IMAGE_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --precision "$PRECISION" \
  --mamba-from-trion "$MAMBA_FROM_TRION" \
  --gradient-checkpointing "$GRADIENT_CHECKPOINTING" \
  --identity-gradient-checkpointing "$IDENTITY_GRADIENT_CHECKPOINTING" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory "$PIN_MEMORY" \
  --limit-val-batches "$LIMIT_VAL_BATCHES" \
  --num-sanity-val-steps "$NUM_SANITY_VAL_STEPS" \
  --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES" \
  --checkpoint-every-n-steps "$CHECKPOINT_EVERY_N_STEPS" \
  --log-dir "$OUT_ROOT"
