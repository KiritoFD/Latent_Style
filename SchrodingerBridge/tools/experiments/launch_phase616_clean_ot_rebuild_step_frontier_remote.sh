#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python tools/experiments/launch_remote_wsl_command.py \
  --task-name phase616_clean_ot_rebuild_step_frontier \
  --remote-workspace-root /mnt/i/Github/Latent_Style \
  --remote-wsl-cwd /mnt/i/Github/Latent_Style/SchrodingerBridge \
  --remote-log-path /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_step_frontier/launcher.log \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py \
  --sync-path SchrodingerBridge/configs/aaai2027/phase616_eval_baseline_dummy055_tau025_stepfrontier.json \
  --sync-path SchrodingerBridge/tools/experiments/run_phase616_clean_ot_rebuild_step_frontier.sh \
  --sync-path SchrodingerBridge/tools/experiments/launch_phase616_clean_ot_rebuild_step_frontier_remote.sh \
  --verify-python-file SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py \
  --health-wait-seconds 15 \
  --max-runtime-memory-mib 11500 \
  -- bash tools/experiments/run_phase616_clean_ot_rebuild_step_frontier.sh
