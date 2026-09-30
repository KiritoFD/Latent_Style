#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 tools/experiments/launch_remote_wsl_command.py \
  --task-name phase618_style_sweep \
  --remote-workspace-root /mnt/i/Github/Latent_Style \
  --remote-wsl-cwd /mnt/i/Github/Latent_Style/SchrodingerBridge \
  --remote-log-path /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/616/logs/phase618_style_sweep.log \
  --host 100.115.18.62 \
  --port 2222 \
  --user administrator \
  --wsl-distro Ubuntu-26.04 \
  --sync-path SchrodingerBridge/src \
  --sync-path SchrodingerBridge/configs/aaai2027 \
  --sync-path SchrodingerBridge/tools/experiments \
  --sync-path SchrodingerBridge/tools/audit_phase618_run_validity.py \
  --sync-path SchrodingerBridge/tools/probe_conditioning_sensitivity.py \
  --sync-path SchrodingerBridge/tools/probe_config_effectiveness.py \
  --sync-path SchrodingerBridge/tools/probe_styleid_eval_path.py \
  --sync-path SchrodingerBridge/tools/probe_training_variant_effect.py \
  --sync-path SchrodingerBridge/docs/616 \
  --sync-path SchrodingerBridge/docs/618 \
  --sync-path SchrodingerBridge/docs/model \
  --sync-path SchrodingerBridge/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json \
  --verify-python-file SchrodingerBridge/src/run.py \
  --verify-python-file SchrodingerBridge/src/losses.py \
  --verify-python-file SchrodingerBridge/src/trainer.py \
  --verify-python-file SchrodingerBridge/src/utils/training.py \
  --verify-python-file SchrodingerBridge/tools/audit_phase618_run_validity.py \
  --verify-python-file SchrodingerBridge/tools/experiments/phase616_auto.py \
  --verify-python-file SchrodingerBridge/tools/probe_conditioning_sensitivity.py \
  --verify-python-file SchrodingerBridge/tools/probe_config_effectiveness.py \
  --verify-python-file SchrodingerBridge/tools/probe_styleid_eval_path.py \
  --verify-python-file SchrodingerBridge/tools/probe_training_variant_effect.py \
  --health-wait-seconds 60 \
  --max-prelaunch-memory-mib 1500 \
  --max-runtime-memory-mib 11570 \
  --runtime-guard-max-memory-mib 11570 \
  --runtime-guard-poll-seconds 10 \
  --runtime-guard-min-memory-mib 0 \
  --runtime-guard-min-mode ignore \
  -- \
  bash tools/experiments/run_phase618_style_sweep.sh "$@"
