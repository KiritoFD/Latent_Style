#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${BASELINE_PYTHON:-/root/venvs/samam/bin/python}
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import torch
assert hasattr(torch, "cuda") and torch.cuda.is_available()
PY
  PYTHON_BIN=$("$SCRIPT_DIR/wsl_find_python_env.sh")
fi

GPU_LOCK_FILE=${GPU_LOCK_FILE:-/mnt/g/GitHub/Latent_Style/SchrodingerBridge/aaai2027/.local_gpu_eval.lock}
WAIT_LOCK_SECONDS=${WAIT_LOCK_SECONDS:-30}
while [[ -f "$GPU_LOCK_FILE" ]]; do
  echo "[run_samam_wikiarts5_segmented_eval_wsl] waiting for local GPU lock: $GPU_LOCK_FILE"
  sleep "$WAIT_LOCK_SECONDS"
done

CONTENT_ROOT=${CONTENT_ROOT:-/mnt/f/wikiarts_5_full_notest/train_flat/content}
STYLE_ROOT=${STYLE_ROOT:-/mnt/f/wikiarts_5_full_notest/train_flat/style}
EVAL_IMAGE_ROOT=${EVAL_IMAGE_ROOT:-/mnt/f/wikiart_distinct5_samam_512_classview/test}
STYLE_NAMES=${STYLE_NAMES:-Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e}
OUT_ROOT=${OUT_ROOT:-/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wikiarts5_patch8_segmented_$(date +%Y%m%d_%H%M%S)}
MAX_STEPS=${MAX_STEPS:-20000}
STOP_AT_MAX_STEPS=${STOP_AT_MAX_STEPS:-0}
HARD_MAX_STEPS=${HARD_MAX_STEPS:-0}
STEP_INTERVAL=${STEP_INTERVAL:-250}
VAL_INTERVAL=${VAL_INTERVAL:-1000}
BATCH_SIZE=${BATCH_SIZE:-1}
TRAIN_IMAGE_SIZE=${TRAIN_IMAGE_SIZE:-256}
TRAIN_CROP_SIZE=${TRAIN_CROP_SIZE:-256}
EVAL_IMAGE_SIZE=${EVAL_IMAGE_SIZE:-256}
PATCH_SIZE=${PATCH_SIZE:-8}
PRECISION=${PRECISION:-32-true}
MAMBA_FROM_TRION=${MAMBA_FROM_TRION:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
IDENTITY_GRADIENT_CHECKPOINTING=${IDENTITY_GRADIENT_CHECKPOINTING:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-0}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-0.1}
NUM_SANITY_VAL_STEPS=${NUM_SANITY_VAL_STEPS:-0}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1}

RESULT_ROOT="$OUT_ROOT"
TRAIN_REPO=/mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam/TRAIN
EVAL_SCRIPT=/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/eval_samam_checkpoint_curve.py
AGG_SCRIPT=/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/aggregate_samam_segmented_curve.py
CONV_SCRIPT=/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/watch_samam_segmented_convergence.py
SUMMARY_JSON="$RESULT_ROOT/segmented_status.json"
SUMMARY_JSONL="$RESULT_ROOT/segmented_status.jsonl"
CONVERGENCE_JSON="$RESULT_ROOT/curve_convergence.json"
CONVERGENCE_PATIENCE=${CONVERGENCE_PATIENCE:-4}
CONVERGENCE_FLAT_EPS_STYLE=${CONVERGENCE_FLAT_EPS_STYLE:-0.006}
CONVERGENCE_FLAT_EPS_LPIPS=${CONVERGENCE_FLAT_EPS_LPIPS:-0.006}
mkdir -p "$RESULT_ROOT"

find_latest_ckpt() {
  find "$RESULT_ROOT" \( -path '*/checkpoints/*.ckpt' -o -path '*/step_checkpoints/*.ckpt' \) -type f | sort | tail -n 1
}

step_of_ckpt() {
  local path="$1"
  local name
  name=$(basename "$path")
  if [[ "$name" =~ step=0*([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$name" =~ step_0*([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo ""
}

write_status() {
  local step="$1"
  local ckpt="$2"
  local phase="$3"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
payload = {
  "step": int($step),
  "checkpoint": r'''$ckpt''',
  "phase": r'''$phase''',
}
Path(r'''$SUMMARY_JSONL''').parent.mkdir(parents=True, exist_ok=True)
with Path(r'''$SUMMARY_JSONL''').open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\\n")
PY
}

refresh_convergence_json() {
  "$PYTHON_BIN" "$CONV_SCRIPT" \
    --root "$RESULT_ROOT" \
    --poll-seconds 1 \
    --max-cycles 1 \
    --output-json "$CONVERGENCE_JSON" \
    --patience "$CONVERGENCE_PATIENCE" \
    --style-key transfer_clip_style \
    --lpips-key transfer_lpips \
    --flat-eps-style "$CONVERGENCE_FLAT_EPS_STYLE" \
    --flat-eps-lpips "$CONVERGENCE_FLAT_EPS_LPIPS"
}

read_converged_flag() {
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

path = Path(r'''$CONVERGENCE_JSON''')
if not path.exists():
    print("false")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print("true" if payload.get("converged") else "false")
PY
}

resume_ckpt=""
start_step="$STEP_INTERVAL"
if existing_ckpt=$(find_latest_ckpt) && [[ -n "$existing_ckpt" ]]; then
  existing_step=$(step_of_ckpt "$existing_ckpt")
  if [[ -n "$existing_step" ]]; then
    resume_ckpt="$existing_ckpt"
    if [[ -f "$RESULT_ROOT/eval_step_$(printf '%06d' "$existing_step")/curve_metrics.csv" ]]; then
      start_step=$(( existing_step + STEP_INTERVAL ))
    else
      start_step=$existing_step
    fi
    echo "[resume] ckpt=$resume_ckpt step=$existing_step start_step=$start_step"
  fi
fi

target_step="$start_step"
while true; do
  if [[ "$HARD_MAX_STEPS" -gt 0 && "$target_step" -gt "$HARD_MAX_STEPS" ]]; then
    echo "[segment] reached hard_max_steps=$HARD_MAX_STEPS without convergence" >&2
    exit 2
  fi
  echo "[segment] target_step=$target_step resume_ckpt=${resume_ckpt:-none}"
  expected_eval_csv="$RESULT_ROOT/eval_step_$(printf '%06d' "$target_step")/curve_metrics.csv"
  latest_ckpt="$resume_ckpt"
  latest_ckpt_step=""
  if [[ -n "$latest_ckpt" ]]; then
    latest_ckpt_step=$(step_of_ckpt "$latest_ckpt")
  fi
  if [[ "$latest_ckpt_step" != "$target_step" || ! -f "$expected_eval_csv" ]]; then
    if [[ "$latest_ckpt_step" != "$target_step" ]]; then
      train_cmd=(
        "$PYTHON_BIN" train_SaMam.py
        --content "$CONTENT_ROOT"
        --style "$STYLE_ROOT"
        --gpus 0
        --iterations "$target_step"
        --val-interval "$VAL_INTERVAL"
        --batch-size "$BATCH_SIZE"
        --train-image-size "$TRAIN_IMAGE_SIZE"
        --train-crop-size "$TRAIN_CROP_SIZE"
        --eval-image-size "$EVAL_IMAGE_SIZE"
        --patch-size "$PATCH_SIZE"
        --precision "$PRECISION"
        --mamba-from-trion "$MAMBA_FROM_TRION"
        --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
        --identity-gradient-checkpointing "$IDENTITY_GRADIENT_CHECKPOINTING"
        --num-workers "$NUM_WORKERS"
        --pin-memory "$PIN_MEMORY"
        --limit-val-batches "$LIMIT_VAL_BATCHES"
        --num-sanity-val-steps "$NUM_SANITY_VAL_STEPS"
        --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES"
        --checkpoint-every-n-steps "$STEP_INTERVAL"
        --log-dir "$RESULT_ROOT"
      )
      if [[ -n "$resume_ckpt" ]]; then
        train_cmd+=(--checkpoint "$resume_ckpt")
      fi
      (
        cd "$TRAIN_REPO"
        "${train_cmd[@]}"
      )
      latest_ckpt=$(find_latest_ckpt)
    fi
  fi
  if [[ -z "$latest_ckpt" ]]; then
    echo "No checkpoint found after target_step=$target_step" >&2
    exit 1
  fi
  write_status "$target_step" "$latest_ckpt" "train_complete"
  eval_root="$RESULT_ROOT/eval_step_$(printf '%06d' "$target_step")"
  if [[ ! -f "$expected_eval_csv" ]]; then
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
      --checkpoint "$latest_ckpt" \
      --image-root "$EVAL_IMAGE_ROOT" \
      --output-root "$eval_root" \
      --image-size 256 \
      --max-src-per-style 30 \
      --metric-batch-size 4 \
      --clip-backend open_clip \
      --style-names "$STYLE_NAMES"
  fi
  "$PYTHON_BIN" "$AGG_SCRIPT" --root "$RESULT_ROOT"
  refresh_convergence_json
  write_status "$target_step" "$latest_ckpt" "eval_complete"
  if [[ "$(read_converged_flag)" == "true" ]]; then
    write_status "$target_step" "$latest_ckpt" "converged_stop"
    echo "[segment] converged at target_step=$target_step"
    break
  fi
  if [[ "$STOP_AT_MAX_STEPS" -eq 1 && "$target_step" -ge "$MAX_STEPS" ]]; then
    echo "[segment] reached max_steps=$MAX_STEPS without convergence"
    break
  fi
  resume_ckpt="$latest_ckpt"
  target_step=$(( target_step + STEP_INTERVAL ))
done
