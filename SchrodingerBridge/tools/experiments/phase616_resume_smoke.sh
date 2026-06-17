#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /mnt/i/.../exp/<batch>/<run_name>" >&2
  exit 2
fi

RUN_DIR="$1"
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
LATEST_CKPT="$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name 'epoch_*.pt' | sort | tail -n 1)"

echo "run_dir=${RUN_DIR}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "cfg=${CFG_PATH}"
echo "ckpt=${LATEST_CKPT}"

cd "${ROOT_DIR}"
timeout 30s python -u src/run.py --config "${CFG_PATH}" --resume "${LATEST_CKPT}"
