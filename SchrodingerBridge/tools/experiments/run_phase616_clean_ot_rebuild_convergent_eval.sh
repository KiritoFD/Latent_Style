#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_convergent_eval"
mkdir -p "${LOG_DIR}"

bash tools/experiments/run_configs_with_gpu_monitor.sh \
  "${LOG_DIR}" \
  "configs/aaai2027/phase616_clean_unbalanced_dummy_vertical_affine_dummy055_tau025_step600_eval_remote.json"
