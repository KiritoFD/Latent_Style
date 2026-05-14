#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"${PYTHON_BIN}" Related_Works/baseline_pipeline/scripts/train_new_baselines.py \
  --baselines artbank \
  --run_root "${RUN_ROOT}" \
  --python "${PYTHON_BIN}" \
  --images_per_style "${IMAGES_PER_STYLE}" \
  --batch_size "${BATCH_SIZE}" \
  --load_size "${LOAD_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --aesfa_iters "${AESFA_ITERS}" \
  --stytr2_iters "${STYTR2_ITERS}" \
  2>&1 | tee "${RUN_ROOT}/logs/launcher_artbank_preflight.log"

echo "[ArtBank] preflight only. Do not start diffusion training until sd-v1-4.ckpt and ArtBank prompt-bank weights are local."
