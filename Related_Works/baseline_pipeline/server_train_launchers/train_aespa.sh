#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

AESPA_REPO="${REPO_ROOT}/Related_Works/AesPA-Net"
AESPA_CKPT="${AESPA_REPO}/baseline_checkpoints/vgg_normalised_conv5_1.t7"
CONTENT_DIR="${RUN_ROOT}/datasets/flat/content"
STYLE_DIR="${RUN_ROOT}/datasets/flat/style"
LOG_PATH="${RUN_ROOT}/logs/aespa_train.log"

"${PYTHON_BIN}" Related_Works/baseline_pipeline/scripts/train_new_baselines.py \
  --baselines prepare-data aespa-net \
  --run_root "${RUN_ROOT}" \
  --python "${PYTHON_BIN}" \
  --images_per_style "${IMAGES_PER_STYLE}" \
  --batch_size "${BATCH_SIZE}" \
  --load_size "${LOAD_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --aesfa_iters "${AESFA_ITERS}" \
  --stytr2_iters "${STYTR2_ITERS}"

if [[ ! -f "${AESPA_CKPT}" ]]; then
  echo "[AesPA-Net] blocked: missing ${AESPA_CKPT}" | tee -a "${LOG_PATH}"
  echo "[AesPA-Net] put the official vgg_normalised_conv5_1.t7 there, then rerun this script." | tee -a "${LOG_PATH}"
  exit 0
fi

mkdir -p "${CONTENT_DIR}" "${STYLE_DIR}" "${RUN_ROOT}/checkpoints/aespa"

"${PYTHON_BIN}" "${AESPA_REPO}/main.py" \
  --type train \
  --comment "server_${VRAM_PROFILE}" \
  --train_result_dir "${RUN_ROOT}/checkpoints/aespa" \
  --content_dir "${CONTENT_DIR}" \
  --style_dir "${STYLE_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --imsize "${LOAD_SIZE}" \
  --cropsize "${CROP_SIZE}" \
  --max_iter "${AESPA_ITERS}" \
  --check_iter "${AESPA_ITERS}" \
  --num_workers "${NUM_WORKERS}" \
  2>&1 | tee -a "${LOG_PATH}"
