#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 tools/experiments/launch_remote_wsl_command.py \
  --task-name phase616_auto_tree_wait_idle \
  --remote-workspace-root /mnt/i/Github/Latent_Style \
  --remote-wsl-cwd /mnt/i/Github/Latent_Style/SchrodingerBridge \
  --remote-log-path /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/616/logs/phase616_auto_tree_wait_idle.log \
  --host 100.115.18.62 \
  --port 2222 \
  --user administrator \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/src \
  --sync-path SchrodingerBridge/configs/aaai2027 \
  --sync-path SchrodingerBridge/tools/experiments \
  --sync-path SchrodingerBridge/docs/616 \
  --verify-python-file SchrodingerBridge/src/run.py \
  --verify-python-file SchrodingerBridge/src/losses.py \
  --verify-python-file SchrodingerBridge/src/trainer.py \
  --verify-python-file SchrodingerBridge/tools/experiments/phase616_auto.py \
  --verify-python-file SchrodingerBridge/tools/experiments/launch_remote_wsl_command.py \
  --health-wait-seconds 60 \
  --max-prelaunch-memory-mib 24576 \
  --max-runtime-memory-mib 11264 \
  --runtime-guard-max-memory-mib 11264 \
  --runtime-guard-poll-seconds 10 \
  --runtime-guard-min-memory-mib 0 \
  --runtime-guard-min-mode ignore \
  --env IDLE_MEM_MIB=1500 \
  --env POLL_SEC=30 \
  -- \
  bash tools/experiments/run_phase616_auto_tree_when_idle.sh "$@"
