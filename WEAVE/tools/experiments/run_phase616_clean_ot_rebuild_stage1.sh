#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_stage1"
mkdir -p "${LOG_DIR}"

bash tools/experiments/run_configs_with_gpu_monitor.sh \
  "${LOG_DIR}" \
  "configs/aaai2027/phase616_clean_ot_probe_tokenentropy_selfaffgw_mix_faststep60_e1_noeval.json" \
  "configs/aaai2027/phase616_clean_vertical_target_selfaffgw_wavelet_faststep60_e1_noeval.json" \
  "configs/aaai2027/phase616_clean_unbalanced_dummy_vertical_affine_faststep60_e1_remote.json"
