#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python tools/experiments/launch_remote_wsl_command.py \
  --task-name phase616_clean_ot_rebuild_eval_frontier \
  --remote-workspace-root /mnt/i/Github/Latent_Style \
  --remote-wsl-cwd /mnt/i/Github/Latent_Style/SchrodingerBridge \
  --remote-log-path /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_eval_frontier/launcher.log \
  --host 100.115.18.62 \
  --port 2222 \
  --user administrator \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/src \
  --sync-path SchrodingerBridge/configs/aaai2027 \
  --sync-path SchrodingerBridge/tools/experiments \
  --sync-path SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616 \
  --verify-python-file SchrodingerBridge/src/run.py \
  --verify-python-file SchrodingerBridge/src/utils/run_evaluation.py \
  --verify-python-file SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py \
  --health-wait-seconds 20 \
  --max-prelaunch-memory-mib 1500 \
  --max-runtime-memory-mib 10800 \
  --runtime-guard-max-memory-mib 10800 \
  --runtime-guard-poll-seconds 10 \
  --runtime-guard-min-memory-mib 0 \
  --runtime-guard-min-mode ignore \
  -- \
  bash tools/experiments/run_phase616_clean_ot_rebuild_eval_frontier.sh
