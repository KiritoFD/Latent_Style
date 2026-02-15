#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/../train_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_watchdog_${STAMP}.log"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-10}"

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

while true; do
  echo "[INFO] launch train at $(date '+%F %T')" | tee -a "${LOG_FILE}"
  python run.py --config config.json 2>&1 | tee -a "${LOG_FILE}"
  EXIT_CODE="${PIPESTATUS[0]}"
  if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "[INFO] training finished normally at $(date '+%F %T')" | tee -a "${LOG_FILE}"
    break
  fi
  echo "[WARN] train crashed with exit=${EXIT_CODE}; restart in ${RESTART_DELAY_SEC}s" | tee -a "${LOG_FILE}"
  sleep "${RESTART_DELAY_SEC}"
done
