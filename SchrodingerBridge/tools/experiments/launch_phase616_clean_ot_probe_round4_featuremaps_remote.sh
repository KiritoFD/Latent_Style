#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python tools/experiments/launch_remote_wsl_command.py \
  --task-name phase616_clean_ot_probe_round4_featuremaps \
  --remote-workspace-root /home/xy/Latent_Style/SchrodingerBridge_phase616 \
  --remote-wsl-cwd /home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge \
  --remote-log-path /home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_probe_round4_featuremaps/launcher.log \
  --host 100.115.18.62 \
  --port 2222 \
  --user administrator \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/src \
  --sync-path SchrodingerBridge/run.py \
  --sync-path SchrodingerBridge/configs/aaai2027 \
  --sync-path SchrodingerBridge/tools/experiments \
  --sync-path SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616 \
  --verify-python-file SchrodingerBridge/src/config_schema.py \
  --verify-python-file SchrodingerBridge/src/losses.py \
  --verify-python-file SchrodingerBridge/src/semantic_tokenizer.py \
  --verify-python-file SchrodingerBridge/src/trainer.py \
  --verify-python-file SchrodingerBridge/src/utils/training.py \
  --health-wait-seconds 30 \
  --max-prelaunch-memory-mib 1500 \
  --max-runtime-memory-mib 11200 \
  --runtime-guard-max-memory-mib 11200 \
  --runtime-guard-poll-seconds 10 \
  -- \
  bash tools/experiments/run_phase616_clean_ot_probe_round4_featuremaps.sh
