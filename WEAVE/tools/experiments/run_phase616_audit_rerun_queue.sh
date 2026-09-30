#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bash tools/experiments/run_phase616_clean_ot_probe_round4_featuremaps.sh
bash tools/experiments/run_phase616_clean_stats_probe_round1.sh
