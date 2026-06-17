#!/usr/bin/env bash
# =============================================================================
# 616 OT Scratch Matched Experiment Launcher
# -----------------------------------------------------------------------------
# Sequentially launches two matched scratch (no warmstart) 24-epoch OT
# experiments on the remote 3060 (12 GB) WSL host:
#
#   1. self_affinity_gw        (pixel-space self-affinity structure cost)
#   2. tokenizer_entropy_affinity_gw (tokenizer-entropy structure cost)
#
# Both share identical OT / bridge / training hyper-parameters except for
# coupling_structure_cost_mode, enabling a controlled structure-cost A/B.
#
# Smoke is skipped because both configs inherit a battle-tested base and the
# OT code path was already exercised in the Round 1-8 fast probes.
#
# Usage:
#   bash tools/experiments/run_phase616_ot_scratch_matched.sh
#
# Optional env overrides:
#   PYTHON_BIN          local Python used to invoke the launcher (default python3)
#   REMOTE_WSL_CWD      remote workspace root (default /mnt/i/Github/Latent_Style)
#   REMOTE_PYTHON       remote venv Python (default /home/xy/venvs/samam312/bin/python)
#   SKIP_SMOKE          1 to skip smoke (default 1 — configs already verified)
#   CONTINUE_ON_FAILURE 1 to continue to the next config if one fails (default 0)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE_WSL_CWD="${REMOTE_WSL_CWD:-/mnt/i/Github/Latent_Style}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/xy/venvs/samam312/bin/python}"
SKIP_SMOKE="${SKIP_SMOKE:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

# ---------------------------------------------------------------------------
# GPU memory guard parameters tuned for the 3060 12 GB.
#   - max-prelaunch 7000 MiB: leave headroom before launching so the health
#     preflight does not fire on residual VRAM from a previous process.
#   - min-runtime 9216 MiB: training needs ~9 GB (b8a2 + bf16 + channels_last).
#   - max-runtime 10800 MiB: hard ceiling; full_eval offload keeps peak below this.
#   - guard-max 11000 MiB: kill if sustained above this for the warmup window.
#   - guard-min-mode warn: configs use full_eval_each_epoch which auto-downgrades
#     "stop" to "warn" inside the launcher; we set it explicitly for clarity.
# ---------------------------------------------------------------------------
MAX_PRELAUNCH_MIB="7000"
MIN_RUNTIME_MIB="9216"
MAX_RUNTIME_MIB="10800"
MIN_RUNTIME_SLACK_MIB="128"
GUARD_MAX_MEMORY_MIB="11000"
GUARD_POLL_SECONDS="10"
GUARD_MIN_MEMORY_MIB="9216"
GUARD_MIN_WARMUP_SECONDS="300"
GUARD_MIN_CONSECUTIVE_POLLS="3"
GUARD_MIN_MODE="warn"
HEALTH_WAIT_SECONDS="30"

# Task prefixes for remote process identification.
PREFIX_SELF_AFFINITY="616_ot_scratch_selfaffgw"
PREFIX_TOK_ENTROPY="616_ot_scratch_tokentropy"

CONFIGS=(
  "${PREFIX_SELF_AFFINITY}|configs/aaai2027/phase616_ot_vertical_scratch_b8a2_e24.json"
  "${PREFIX_TOK_ENTROPY}|configs/aaai2027/phase616_ot_tokentropy_scratch_b8a2_e24.json"
)

smoke_args=()
if [[ "${SKIP_SMOKE}" == "1" ]]; then
  smoke_args+=(--skip-smoke)
fi

overall_rc=0
for entry in "${CONFIGS[@]}"; do
  task_prefix="${entry%%|*}"
  config="${entry##*|}"
  echo ""
  echo "================================================================"
  echo "[phase616_ot_scratch_matched] launching ${task_prefix}"
  echo "  config:        ${config}"
  echo "  remote_cwd:    ${REMOTE_WSL_CWD}"
  echo "  remote_python: ${REMOTE_PYTHON}"
  echo "  skip_smoke:    ${SKIP_SMOKE}"
  echo "================================================================"

  set +e
  "${PYTHON_BIN}" tools/experiments/launch_remote_experiment_train.py \
    --config "${config}" \
    --remote-wsl-cwd "${REMOTE_WSL_CWD}" \
    --remote-python "${REMOTE_PYTHON}" \
    --max-prelaunch-memory-mib "${MAX_PRELAUNCH_MIB}" \
    --min-runtime-memory-mib "${MIN_RUNTIME_MIB}" \
    --max-runtime-memory-mib "${MAX_RUNTIME_MIB}" \
    --min-runtime-slack-mib "${MIN_RUNTIME_SLACK_MIB}" \
    --runtime-guard-max-memory-mib "${GUARD_MAX_MEMORY_MIB}" \
    --runtime-guard-poll-seconds "${GUARD_POLL_SECONDS}" \
    --runtime-guard-min-memory-mib "${GUARD_MIN_MEMORY_MIB}" \
    --runtime-guard-min-warmup-seconds "${GUARD_MIN_WARMUP_SECONDS}" \
    --runtime-guard-min-consecutive-polls "${GUARD_MIN_CONSECUTIVE_POLLS}" \
    --runtime-guard-min-mode "${GUARD_MIN_MODE}" \
    --health-wait-seconds "${HEALTH_WAIT_SECONDS}" \
    --task-prefix "${task_prefix}" \
    "${smoke_args[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[phase616_ot_scratch_matched] WARNING: ${task_prefix} exited with rc=${rc}"
    overall_rc="${rc}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      echo "[phase616_ot_scratch_matched] aborting sequential run (set CONTINUE_ON_FAILURE=1 to proceed)"
      exit "${rc}"
    fi
  else
    echo "[phase616_ot_scratch_matched] ${task_prefix} completed successfully"
  fi
done

echo ""
if [[ "${overall_rc}" -eq 0 ]]; then
  echo "[phase616_ot_scratch_matched] all configs completed successfully"
else
  echo "[phase616_ot_scratch_matched] finished with errors (overall_rc=${overall_rc})"
fi
exit "${overall_rc}"
