#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${REPO_ROOT}/src"
LOCK_FILE="/tmp/latent_style_overfit50_v2.lock"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Single instance guard.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another overfit50_v2 runner is active (lock: ${LOCK_FILE})."
  exit 1
fi

# name + config path (relative to src/)
EXPERIMENT_MATRIX=(
  "overfit50_e1_baseline_d2_ref_fixfull experiments/overfit50_e1_baseline_d2_ref_fixfull.json"
  "overfit50_e3_highpass_strong experiments/overfit50_e3_highpass_strong.json"
  "overfit50_e9_hifeat_probgate_v1 experiments/overfit50_e9_hifeat_probgate_v1.json"
)

cleanup() {
  pkill -f "python .*run.py --config" >/dev/null 2>&1 || true
  pkill -f "python .*utils/run_small_experiment.py --config" >/dev/null 2>&1 || true
}

run_experiment() {
  local name="$1"
  local cfg="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${name}_${ts}.log"

  cleanup
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
  trap cleanup EXIT
  for item in "${EXPERIMENT_MATRIX[@]}"; do
    local name="${item%% *}"
    local cfg="${item#* }"
    run_experiment "${name}" "${cfg}"
  done
  echo "All overfit50_v2 experiments completed."
}

main
