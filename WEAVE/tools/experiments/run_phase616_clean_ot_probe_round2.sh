#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_probe_round2"
mkdir -p "$LOG_DIR"

bash tools/experiments/run_configs_with_gpu_monitor.sh \
  "$LOG_DIR" \
  "configs/aaai2027/phase616_clean_ot_probe_selfaffgw_faststep60_e1.json" \
  "configs/aaai2027/phase616_clean_ot_probe_lowedge_selfaffgw_faststep60_e1.json"
