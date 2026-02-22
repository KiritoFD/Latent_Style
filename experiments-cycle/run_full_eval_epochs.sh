#!/usr/bin/env bash
set -euo pipefail

# Batch full evaluation for selected checkpoints in THIS repository.
#
# Example:
#   bash run_full_eval_epochs.sh "50 100 150 200"
#   CHECKPOINT_DIR=./full_300-map16+32 bash run_full_eval_epochs.sh "150 200 250 300"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
SRC_DIR="${REPO_ROOT}/src"

EPOCHS="${1:-50 100 150 200}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/full_300-map16+32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_DIR}/full_eval_manual}"
TEST_DIR="${TEST_DIR:-${REPO_ROOT}/../style_data/overfit50}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/eval_cache}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${REPO_ROOT}/style_classifier.pt}"
NUM_STEPS="${NUM_STEPS:-3}"
STEP_SIZE="${STEP_SIZE:-1.15}"
STYLE_STRENGTH="${STYLE_STRENGTH:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_SRC_SAMPLES="${MAX_SRC_SAMPLES:-50}"
MAX_REF_COMPARE="${MAX_REF_COMPARE:-50}"
MAX_REF_CACHE="${MAX_REF_CACHE:-100}"
REF_FEATURE_BATCH_SIZE="${REF_FEATURE_BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EVAL_SCRIPT="${SRC_DIR}/utils/run_evaluation.py"

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

  (
    cd "${SRC_DIR}"
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --checkpoint "${ckpt}" \
      --output "${out_dir}" \
      --test_dir "${TEST_DIR}" \
      --cache_dir "${CACHE_DIR}" \
      --num_steps "${NUM_STEPS}" \
      --step_size "${STEP_SIZE}" \
      --style_strength "${STYLE_STRENGTH}" \
      --batch_size "${BATCH_SIZE}" \
      --max_src_samples "${MAX_SRC_SAMPLES}" \
      --max_ref_compare "${MAX_REF_COMPARE}" \
      --max_ref_cache "${MAX_REF_CACHE}" \
      --ref_feature_batch_size "${REF_FEATURE_BATCH_SIZE}" \
      --classifier_path "${CLASSIFIER_CKPT}" \
      --save_pairs \
      > "${log_file}" 2>&1
  )

  echo "[DONE] epoch=${epoch}, log=${log_file}"
done

echo "All requested epochs processed."
