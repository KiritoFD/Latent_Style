#!/usr/bin/env bash
set -euo pipefail
LOCK_FILE="/tmp/latent_style_full_eval.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another full-eval batch is running (lock: ${LOCK_FILE})."
  exit 1
fi

# Batch full evaluation for specific checkpoints.
# Default epochs: 50 100 150 200
#
# Usage:
#   bash NCE_SWD/run_full_eval_epochs.sh
#   bash NCE_SWD/run_full_eval_epochs.sh "50 100 150 200"
#   CHECKPOINT_DIR=/path/to/ckpts OUTPUT_ROOT=/path/to/out bash NCE_SWD/run_full_eval_epochs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EPOCHS="${1:-50 100 150 200}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/NCE_SWD/adacut}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_DIR}/full_eval_manual}"
TEST_DIR="${TEST_DIR:-${REPO_ROOT}/style_data/test}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/eval_cache}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${REPO_ROOT}/Thermal/src/style_classifier.pt}"
NUM_STEPS="${NUM_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MAX_SRC_SAMPLES="${MAX_SRC_SAMPLES:-50}"
MAX_REF_COMPARE="${MAX_REF_COMPARE:-50}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_SCRIPT="${REPO_ROOT}/NCE_SWD/src/utils/run_evaluation.py"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: evaluation script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Repo root      : ${REPO_ROOT}"
echo "Checkpoint dir : ${CHECKPOINT_DIR}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "Epochs         : ${EPOCHS}"

for epoch in ${EPOCHS}; do
  ckpt="${CHECKPOINT_DIR}/epoch_$(printf "%04d" "${epoch}").pt"
  out_dir="${OUTPUT_ROOT}/epoch_$(printf "%04d" "${epoch}")"
  log_file="${OUTPUT_ROOT}/eval_epoch_$(printf "%04d" "${epoch}").log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[SKIP] checkpoint not found: ${ckpt}"
    continue
  fi

  mkdir -p "${out_dir}"
  echo "[RUN ] epoch=${epoch} -> ${out_dir}"

  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${ckpt}" \
    --output "${out_dir}" \
    --test_dir "${TEST_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_src_samples "${MAX_SRC_SAMPLES}" \
    --max_ref_compare "${MAX_REF_COMPARE}" \
    --classifier_path "${CLASSIFIER_CKPT}" \
    > "${log_file}" 2>&1

  echo "[DONE] epoch=${epoch}, log=${log_file}"
done

echo "All requested epochs processed."
