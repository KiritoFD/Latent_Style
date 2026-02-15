#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/../train_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_watchdog_${STAMP}.log"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-10}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/config.json}"

cd "${ROOT_DIR}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[FATAL] conda.sh not found" | tee -a "${LOG_FILE}"
  exit 1
fi

conda activate cu128

echo "[INFO] watchdog started at $(date '+%F %T')" | tee -a "${LOG_FILE}"
echo "[INFO] cwd=${ROOT_DIR}" | tee -a "${LOG_FILE}"
echo "[INFO] log=${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "[INFO] config=${CONFIG_PATH}" | tee -a "${LOG_FILE}"
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[FATAL] config not found: ${CONFIG_PATH}" | tee -a "${LOG_FILE}"
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  CONFIG_SHA="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
  echo "[INFO] config_sha256=${CONFIG_SHA}" | tee -a "${LOG_FILE}"
fi
echo "[INFO] config_effective_snippet:" | tee -a "${LOG_FILE}"
grep -E '"(batch_size|use_amp|amp_dtype|use_grad_scaler|channels_last|strict_batch_sanity|cuda_sync_debug|preload_to_gpu|use_compile|resume_checkpoint)"' "${CONFIG_PATH}" | tee -a "${LOG_FILE}" || true

while true; do
  echo "[INFO] launch train at $(date '+%F %T')" | tee -a "${LOG_FILE}"
  python run.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_FILE}"
  EXIT_CODE="${PIPESTATUS[0]}"
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "[INFO] training finished normally at $(date '+%F %T')" | tee -a "${LOG_FILE}"
    break
  fi
  echo "[WARN] train crashed with exit=${EXIT_CODE}; restart in ${RESTART_DELAY_SEC}s" | tee -a "${LOG_FILE}"
  sleep "${RESTART_DELAY_SEC}"
done
