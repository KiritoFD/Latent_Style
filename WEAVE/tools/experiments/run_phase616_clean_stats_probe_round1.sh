#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_stats_probe_round1"
BUILD_BANK="${BUILD_BANK:-1}"
BANK_CONFIG="configs/aaai2027/phase616_clean_stats_probe_control_none_faststep60_e1.json"

mkdir -p "$LOG_DIR"

if [[ "$BUILD_BANK" == "1" ]]; then
  echo "[phase616_clean_stats_probe_round1] building style stats bank from ${BANK_CONFIG}"
  "${PYTHON_BIN}" tools/experiments/build_phase616_style_stats_bank.py --config "${BANK_CONFIG}" | tee "${LOG_DIR}/style_stats_bank_build.json"
fi

bash tools/experiments/run_configs_with_gpu_monitor.sh \
  "$LOG_DIR" \
  "configs/aaai2027/phase616_clean_stats_probe_control_none_faststep60_e1.json" \
  "configs/aaai2027/phase616_clean_stats_probe_terminal_affine_faststep60_e1.json" \
  "configs/aaai2027/phase616_clean_stats_probe_normalized_solver_faststep60_e1.json"
