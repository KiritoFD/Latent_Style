#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/latent_style_overfit_experiments.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another overfit experiment script is running (lock: ${LOCK_FILE})."
  exit 1
fi

EXPERIMENT_MATRIX=(
  "overfit50_e1_baseline_d2_ref experiments/overfit50_e1_baseline_d2_ref.json"
  "overfit50_e2_highpass_default experiments/overfit50_e2_highpass_default.json"
  "overfit50_e3_highpass_strong experiments/overfit50_e3_highpass_strong.json"
  "overfit50_e4_highpass_light experiments/overfit50_e4_highpass_light.json"
)

cleanup() {
  pkill -f "run_small_experiment.py --config overfit50.json" >/dev/null 2>&1 || true
  pkill -f "python run.py --config" >/dev/null 2>&1 || true
}

run_experiment() {
  local name="$1"
  local cfg="$2"
  cleanup
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export TORCH_JIT=0
  echo "=== Running ${name} ==="
  /bin/bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate cu128 && cd ${REPO_ROOT}/src && python utils/run_small_experiment.py --config ${cfg} --name ${name} --epochs 8 --max_src_samples 50 --batch_size 100 --per_pair 3"
}

main() {
  trap cleanup EXIT
  for item in "${EXPERIMENT_MATRIX[@]}"; do
    name="${item%% *}"
    cfg="${item#* }"
    run_experiment "${name}" "${cfg}"
  done
}

main
