#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python tools/experiments/launch_remote_wsl_command.py \
  --task-name phase616_clean_ot_rebuild_terminalaffine_step400_eval \
  --remote-workspace-root /mnt/i/Github/Latent_Style \
  --remote-wsl-cwd /mnt/i/Github/Latent_Style/SchrodingerBridge \
  --remote-log-path /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_terminalaffine_step400_eval/launcher.log \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/configs/aaai2027/phase616_clean_unbalanced_dummy_vertical_affine_terminalaffine_dummy055_tau025_step400_eval_remote.json \
  --sync-path SchrodingerBridge/tools/experiments/run_phase616_clean_ot_rebuild_terminalaffine_step400_eval.sh \
  --sync-path SchrodingerBridge/tools/experiments/launch_phase616_clean_ot_rebuild_terminalaffine_step400_eval_remote.sh \
  --health-wait-seconds 30 \
  --min-runtime-memory-mib 4000 \
  --max-runtime-memory-mib 11500 \
  -- bash tools/experiments/run_phase616_clean_ot_rebuild_terminalaffine_step400_eval.sh
