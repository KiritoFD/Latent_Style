#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /mnt/i/.../exp/<batch>/<run_name> [extra run.py args...]" >&2
  exit 2
fi

RUN_DIR="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_artifact_dir() {
  local logical_dir="$1"
  if [[ -d "${logical_dir}/logs" || -d "${logical_dir}/full_eval_transfer" ]]; then
    printf '%s\n' "${logical_dir}"
    return 0
  fi
  local nested="${ROOT_DIR}/${logical_dir#/}"
  if [[ -d "${nested}/logs" || -d "${nested}/full_eval_transfer" ]]; then
    printf '%s\n' "${nested}"
    return 0
  fi
  printf '%s\n' "${logical_dir}"
}

ARTIFACT_DIR="$(resolve_artifact_dir "${RUN_DIR}")"
CFG_PATH="${RUN_DIR}/config.json"
RUN_NAME="$(basename "${RUN_DIR}")"
RUN_BATCH_NAME="$(basename "$(dirname "${RUN_DIR}")")"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "config not found: ${CFG_PATH}" >&2
  exit 3
fi

LATEST_CKPT="$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name 'epoch_*.pt' | sort | tail -n 1)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "no checkpoint found in ${ARTIFACT_DIR}" >&2
  exit 4
fi

pkill -f "${RUN_BATCH_NAME}/${RUN_NAME}/config.json" || true
pkill -f "${RUN_NAME}/config.json" || true
pkill -f "launch_all.sh" || true
sleep 2

cd "${ROOT_DIR}"
LOG_PATH="${ARTIFACT_DIR}/resume.out"
nohup python src/run.py --config "${CFG_PATH}" --resume "${LATEST_CKPT}" "$@" >"${LOG_PATH}" 2>&1 &
NEW_PID=$!

echo "run_dir=${RUN_DIR}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "checkpoint=${LATEST_CKPT}"
echo "pid=${NEW_PID}"
echo "log=${LOG_PATH}"
