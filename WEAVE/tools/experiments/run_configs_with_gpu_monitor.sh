#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <log_dir> <config1> [config2 ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$1"
shift
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-2.0}"
MONITOR_GPU_INDEX="${MONITOR_GPU_INDEX:-0}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

for config in "$@"; do
  stem="$(basename "${config}" .json)"
  log_path="${LOG_DIR}/${stem}.log"
  gpu_csv="${LOG_DIR}/${stem}.gpu_metrics.csv"
  gpu_json="${LOG_DIR}/${stem}.gpu_summary.json"
  echo "[run_configs_with_gpu_monitor] starting ${config}" | tee "${log_path}"
  stdbuf -oL -eL "${PYTHON_BIN}" src/run.py --config "${config}" > >(tee -a "${log_path}") 2> >(tee -a "${log_path}" >&2) &
  train_pid=$!
  stdbuf -oL -eL "${PYTHON_BIN}" tools/experiments/monitor_pid_gpu.py \
    --pid "${train_pid}" \
    --csv-out "${gpu_csv}" \
    --summary-out "${gpu_json}" \
    --interval-sec "${MONITOR_INTERVAL_SEC}" \
    --gpu-index "${MONITOR_GPU_INDEX}" > >(tee -a "${log_path}") 2> >(tee -a "${log_path}" >&2) &
  monitor_pid=$!
  set +e
  wait "${train_pid}"
  status=$?
  set -e
  wait "${monitor_pid}" || true
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done
