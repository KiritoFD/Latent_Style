#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/i/Github/Latent_Style}"
PYTHON_BIN="${PYTHON_BIN:-/home/xy/venvs/samam312/bin/python}"
REMOTE_SB="${REMOTE_ROOT}/SchrodingerBridge"
TEST_DIR="${TEST_DIR:-/mnt/i/wikiart_distinct5_samam_512_classview/test}"
CACHE_DIR="${CACHE_DIR:-${REMOTE_ROOT}/eval_cache}"
CLIP_HF_CACHE_DIR="${CLIP_HF_CACHE_DIR:-${CACHE_DIR}/hf}"
FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-16}"
FULL_EVAL_VAE_DECODE_BATCH_SIZE="${FULL_EVAL_VAE_DECODE_BATCH_SIZE:-16}"
IDT_CLIP_STYLE="${IDT_CLIP_STYLE:-0.639920825263}"

checkpoint="${REMOTE_ROOT}/exp/620_spatial_bridge/620_swd12_sigma002_nfe8_b80/epoch_0008.pt"
run_prefix="620_swd12_epoch0008"
nfe_list="4 8 16"
sigma_list="0.0 0.02"
force_regen="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint)
      checkpoint="$2"
      shift 2
      ;;
    --run-prefix)
      run_prefix="$2"
      shift 2
      ;;
    --nfe-list)
      nfe_list="$2"
      shift 2
      ;;
    --sigma-list)
      sigma_list="$2"
      shift 2
      ;;
    --no-force-regen)
      force_regen="0"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$REMOTE_ROOT"
export PYTHONPATH="${REMOTE_SB}/src:${REMOTE_SB}/tools:${REMOTE_SB}/tools/experiments:${PYTHONPATH:-}"

test -f "$checkpoint"
test -d "$TEST_DIR"
mkdir -p "${REMOTE_ROOT}/exp/620_spatial_bridge"

if [ $((FULL_EVAL_BATCH_SIZE % 16)) -ne 0 ]; then
  echo "FULL_EVAL_BATCH_SIZE must be divisible by 16, got ${FULL_EVAL_BATCH_SIZE}" >&2
  exit 2
fi
if [ $((FULL_EVAL_VAE_DECODE_BATCH_SIZE % 16)) -ne 0 ]; then
  echo "FULL_EVAL_VAE_DECODE_BATCH_SIZE must be divisible by 16, got ${FULL_EVAL_VAE_DECODE_BATCH_SIZE}" >&2
  exit 2
fi

sigma_tag() {
  local raw="$1"
  python - "$raw" <<'PY'
import sys
value = float(sys.argv[1])
if abs(value) < 1e-12:
    print("sigma000")
else:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    whole, _, frac = text.partition(".")
    print(f"sigma{whole}{frac}")
PY
}

run_eval() {
  local nfe="$1"
  local sigma="$2"
  local tag
  tag="$(sigma_tag "$sigma")"
  local run_name="${run_prefix}_nfe${nfe}_${tag}"
  local run_dir="${REMOTE_ROOT}/exp/620_spatial_bridge/${run_name}"
  local out_dir="${run_dir}/full_eval/epoch_0008"
  local override="${REMOTE_SB}/configs/_generated_${run_name}_eval_override.json"
  mkdir -p "$run_dir" "$(dirname "$override")"
  RUN_NAME="$run_name" NFE="$nfe" SIGMA="$sigma" OVERRIDE="$override" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "bridge": {"bridge_sigma": float(os.environ["SIGMA"])},
    "inference": {"num_steps": int(os.environ["NFE"])},
    "full_eval": {
        "num_steps": int(os.environ["NFE"]),
        "clip_style_idt_baseline": 0.639920825263,
    },
    "ablation": {
        "name": os.environ["RUN_NAME"],
        "axis": "620_eval_sweep",
        "stage": "eval_only",
    },
}
Path(os.environ["OVERRIDE"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(os.environ["OVERRIDE"])
PY
  echo "[620-eval] run=${run_name} checkpoint=${checkpoint} nfe=${nfe} sigma=${sigma}"
  eval_args=(
    --checkpoint "$checkpoint"
    --output "$out_dir"
    --config_override "$override"
    --test_dir "$TEST_DIR"
    --cache_dir "$CACHE_DIR"
    --clip_hf_cache_dir "$CLIP_HF_CACHE_DIR"
    --num_steps "$nfe"
    --batch_size "$FULL_EVAL_BATCH_SIZE"
    --target_chunk_size 2
    --vae_decode_batch_size "$FULL_EVAL_VAE_DECODE_BATCH_SIZE"
    --vae_compile_method pt2
    --vae_compile_mode reduce-overhead
    --skip_diffusers_vae_when_onnx
    --clip_style_idt_baseline "$IDT_CLIP_STYLE"
    --eval_lpips_chunk_size 4
    --postprocess_mode none
    --postprocess_strength 0.0
    --postprocess_mean_strength 1.0
    --postprocess_std_strength 1.0
    --postprocess_ref_limit 64
    --latent_postprocess_mode none
    --latent_postprocess_strength 0.0
    --latent_postprocess_mean_strength 1.0
    --latent_postprocess_std_strength 1.0
    --latent_postprocess_ref_limit 64
    --no-eval_enable_introstyle
    --introstyle_modelscope_id stabilityai/stable-diffusion-2-1-base
    --introstyle_bank_limit_per_style 64
    --introstyle_batch_size 4
    --introstyle_topk 8
    --introstyle_t 25
    --introstyle_up_ft_index 1
    --introstyle_ensemble_size 1
    --no-save_generated_images
    --no-save_summary_grid
    --keep_generated_on_device
    --source_latent_cache
    --profile_timing
    --no-eval_enable_art_fid
    --no-eval_enable_kid
  )
  if [ "$force_regen" = "1" ]; then
    eval_args+=(--force_regen)
  fi
  "$PYTHON_BIN" SchrodingerBridge/src/utils/run_evaluation.py "${eval_args[@]}"
  "$PYTHON_BIN" SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py \
    --run-dir "$run_dir" \
    --eval-subdir full_eval
}

declare -A seen=()
for nfe in $nfe_list; do
  key="${nfe}|0.02"
  if [ -z "${seen[$key]+x}" ]; then
    seen[$key]=1
    run_eval "$nfe" "0.02"
  fi
done

for sigma in $sigma_list; do
  key="8|${sigma}"
  if [ -z "${seen[$key]+x}" ]; then
    seen[$key]=1
    run_eval "8" "$sigma"
  fi
done

echo "[620-eval] complete"
