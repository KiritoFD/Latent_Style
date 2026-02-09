#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${REPO_ROOT}/logs"
LOCK_FILE="/tmp/latent_style_overfit50_stylepush.lock"
mkdir -p "${LOG_DIR}"

DO_CLEAR=0
for arg in "$@"; do
  case "$arg" in
    --clear) DO_CLEAR=1 ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $(basename "$0") [--clear]"
      exit 1
      ;;
  esac
done

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another overfit50 stylepush runner is active (lock: ${LOCK_FILE})."
  exit 1
fi

EXPERIMENT_MATRIX=(
  "overfit50_e12_hires6_hifeat_v1 experiments/overfit50_e12_hires6_hifeat_v1.json"
  "overfit50_e13_hires6_spatialproto_v1 experiments/overfit50_e13_hires6_spatialproto_v1.json"
  "overfit50_e14_hires6_weakcls_v1 experiments/overfit50_e14_hires6_weakcls_v1.json"
)

cleanup_proc() {
  pkill -f "python .*run.py --config" >/dev/null 2>&1 || true
  pkill -f "python .*utils/run_small_experiment.py --config" >/dev/null 2>&1 || true
  pkill -f "python3.12 .*run.py --config" >/dev/null 2>&1 || true
  pkill -f "python3.12 .*utils/run_small_experiment.py --config" >/dev/null 2>&1 || true
}

run_experiment() {
  local name="$1"
  local cfg="$2"
  local ts log_file
  mkdir -p "${LOG_DIR}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/${name}_${ts}.log"

  cleanup_proc
  sleep 2

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export TORCH_JIT=0

  echo "=== Running ${name} (${cfg}) ==="
  echo "log -> ${log_file}"

  /bin/bash -lc "
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate cu128 && \
    cd ${SRC_DIR} && \
    python utils/run_small_experiment.py \
      --config ${cfg} \
      --name ${name} \
      --epochs 8 \
      --max_src_samples 50 \
      --batch_size 50 \
      --per_pair 3
  " >"${log_file}" 2>&1
}

main() {
  trap cleanup_proc EXIT
  if [[ "${DO_CLEAR}" -eq 1 ]]; then
    echo "Clearing previous artifacts (move mode)..."
    /bin/bash "${REPO_ROOT}/clear.sh" --move --kill
    mkdir -p "${LOG_DIR}"
  fi
  for item in "${EXPERIMENT_MATRIX[@]}"; do
    local name="${item%% *}"
    local cfg="${item#* }"
    run_experiment "${name}" "${cfg}"
  done
  echo "All stylepush overfit50 experiments completed."
}

main
