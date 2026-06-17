#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical/h0_vertical_fm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION_NAME="phase616_h0_resume"

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
RESUME_SCRIPT="${SCRIPT_DIR}/phase616_resume_training_state_foreground.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux missing" >&2
  exit 5
fi

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash '${RESUME_SCRIPT}' '${RUN_DIR}'"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
sleep 2
tmux list-sessions
echo "artifact_dir=${ARTIFACT_DIR}"
echo "session=${SESSION_NAME}"
