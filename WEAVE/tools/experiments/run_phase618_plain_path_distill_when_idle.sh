#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IDLE_MEM_MIB="${IDLE_MEM_MIB:-1500}"
POLL_SEC="${POLL_SEC:-30}"
WSL_GPU_SMI="$(command -v nvidia-smi || true)"
if [[ -z "${WSL_GPU_SMI}" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  WSL_GPU_SMI="/usr/lib/wsl/lib/nvidia-smi"
fi

gpu_used_mib() {
  if [[ -z "${WSL_GPU_SMI}" ]]; then
    echo 0
    return 0
  fi
  "${WSL_GPU_SMI}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{m=0}{v=int($1+0); if(v>m)m=v}END{print m}'
}

active_train_count() {
  ps -eo args | awk '/[s]rc\/run\.py/ {count++} END{print count+0}'
}

echo "[phase618_plain_path_distill_idle_queue] start $(date -Iseconds) cwd=${ROOT_DIR}"
echo "[phase618_plain_path_distill_idle_queue] idle_mem_mib=${IDLE_MEM_MIB} poll_sec=${POLL_SEC}"

while true; do
  train_count="$(active_train_count)"
  used_mib="$(gpu_used_mib)"
  echo "[phase618_plain_path_distill_idle_queue] $(date -Iseconds) active_train=${train_count} gpu_used_mib=${used_mib}"
  if [[ "${train_count}" == "0" && "${used_mib}" -le "${IDLE_MEM_MIB}" ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[phase618_plain_path_distill_idle_queue] idle reached at $(date -Iseconds); launching plain-path distill"
exec bash tools/experiments/run_phase618_plain_path_distill.sh "$@"
