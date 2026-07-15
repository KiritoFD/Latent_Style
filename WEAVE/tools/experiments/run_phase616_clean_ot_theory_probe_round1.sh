#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_theory_probe_round1"
mkdir -p "$LOG_DIR"

bash tools/experiments/run_configs_with_gpu_monitor.sh \
  "$LOG_DIR" \
  "configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix050_warmtopo_k085e3_b16a1_vlen010_e6.json" \
  "configs/aaai2027/phase616_clean_ot_probe_tokenentropy_mix050_warmtopo_k085e3_b16a1_vlen010_e6.json"
