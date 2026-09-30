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
RESUME_CFG="${ARTIFACT_DIR}/config.resume_state.json"
LATEST_CKPT="$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name 'epoch_*.pt' | sort | tail -n 1)"

python - <<'PY' "${CFG_PATH}" "${RESUME_CFG}"
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
training = cfg.setdefault("training", {})
training["resume_optimizer"] = True
training["resume_training_state"] = True
training["resume_prefer_local_checkpoint"] = False
training["full_eval_each_epoch"] = True
training["full_eval_defer_until_training_end"] = False
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

echo "run_dir=${RUN_DIR}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "resume_cfg=${RESUME_CFG}"
echo "ckpt=${LATEST_CKPT}"

cd "${ROOT_DIR}"
timeout 30s python -u src/run.py --config "${RESUME_CFG}" --resume "${LATEST_CKPT}"
