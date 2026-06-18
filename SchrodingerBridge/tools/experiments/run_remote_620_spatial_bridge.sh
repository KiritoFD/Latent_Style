#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/i/Github/Latent_Style}"
PYTHON_BIN="${PYTHON_BIN:-/home/xy/venvs/samam312/bin/python}"
REMOTE_SB="${REMOTE_ROOT}/SchrodingerBridge"
LATENT_ROOT="${LATENT_ROOT:-/mnt/i/wikiart_distinct5_samam_512_latents_ema/train}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/i/wikiart_distinct5_samam_512_classview/train}"
DINO_CACHE="${DINO_CACHE:-${REMOTE_ROOT}/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt}"
PAIRING_PLAN="${PAIRING_PLAN:-${LATENT_ROOT}/.latent_cache/dino_pairing_top8.pt}"
STYLES="${STYLES:-Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e}"
DINO_ALLOW_NETWORK="${DINO_ALLOW_NETWORK:-0}"
DINO_HF_CACHE_DIR="${DINO_HF_CACHE_DIR:-/mnt/i/hf_cache}"
BATCH_SIZE_WAS_SET="${BATCH_SIZE+x}"
BATCH_SIZE="${BATCH_SIZE:-80}"
FULL_EVAL_BATCH_SIZE="${FULL_EVAL_BATCH_SIZE:-16}"
FULL_EVAL_VAE_DECODE_BATCH_SIZE="${FULL_EVAL_VAE_DECODE_BATCH_SIZE:-16}"

variant="base"
epochs="1"
formal="0"
run_name=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant)
      variant="$2"
      shift 2
      ;;
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --formal)
      formal="1"
      shift
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      BATCH_SIZE_WAS_SET=1
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$variant" in
  base)
    config_rel="SchrodingerBridge/configs/620_spatial_bridge_base.json"
    default_run_name="620_base_swd8_sigma002_nfe8_b80"
    ;;
  swd4)
    config_rel="SchrodingerBridge/configs/620_spatial_bridge_swd4.json"
    default_run_name="620_swd4_sigma002_nfe8_b80"
    ;;
  swd12)
    config_rel="SchrodingerBridge/configs/620_spatial_bridge_swd12.json"
    default_run_name="620_swd12_sigma002_nfe8_b80"
    ;;
  adapter)
    config_rel="SchrodingerBridge/configs/620_spatial_bridge_adapter.json"
    default_run_name="620_adapter_swd12_sigma002_nfe8_b64"
    if [ -z "$BATCH_SIZE_WAS_SET" ]; then
      BATCH_SIZE="64"
    fi
    ;;
  *)
    echo "invalid --variant: $variant" >&2
    exit 2
    ;;
esac

if [ "$formal" = "1" ]; then
  epochs="8"
  stage="formal"
else
  stage="smoke"
fi

if [ -z "$run_name" ]; then
  run_name="$default_run_name"
fi

cd "$REMOTE_ROOT"
export PYTHONPATH="${REMOTE_SB}/src:${REMOTE_SB}/tools:${REMOTE_SB}/tools/experiments:${PYTHONPATH:-}"
export REMOTE_ROOT REMOTE_SB LATENT_ROOT IMAGE_ROOT DINO_CACHE PAIRING_PLAN STYLES
export DINO_ALLOW_NETWORK DINO_HF_CACHE_DIR
export BATCH_SIZE FULL_EVAL_BATCH_SIZE FULL_EVAL_VAE_DECODE_BATCH_SIZE
export CONFIG_REL="$config_rel"
export RUN_NAME="$run_name"
export STAGE="$stage"
export EPOCHS="$epochs"

echo "[620] remote_root=${REMOTE_ROOT}"
echo "[620] variant=${variant} stage=${stage} epochs=${epochs}"
echo "[620] dataset root: ${LATENT_ROOT}"
test -d "$LATENT_ROOT"
test -d "$IMAGE_ROOT"

"$PYTHON_BIN" - <<PY
import os
from pathlib import Path

root = Path(os.environ["LATENT_ROOT"])
styles = [x.strip() for x in os.environ["STYLES"].split(",") if x.strip()]
counts = {style: len([p for p in (root / style).iterdir() if p.is_file()]) for style in styles}
print("[620] balanced_counts", counts)
if any(count != 1000 for count in counts.values()):
    raise SystemExit(f"expected exactly 1000 train latents per style, got {counts}")
PY

if [ ! -f "$DINO_CACHE" ]; then
  echo "[620] building DINO cache: ${DINO_CACHE}"
  mkdir -p "$(dirname "$DINO_CACHE")"
  dino_cache_args=(
    --latent-root "$LATENT_ROOT"
    --image-root "$IMAGE_ROOT"
    --output "$DINO_CACHE"
  )
  if [ -n "$DINO_HF_CACHE_DIR" ]; then
    mkdir -p "$DINO_HF_CACHE_DIR"
    dino_cache_args+=(--hf-cache-dir "$DINO_HF_CACHE_DIR")
  fi
  case "${DINO_ALLOW_NETWORK,,}" in
    1|true|yes|on)
      dino_cache_args+=(--allow-network)
      ;;
  esac
  echo "[620] DINO_ALLOW_NETWORK=${DINO_ALLOW_NETWORK} DINO_HF_CACHE_DIR=${DINO_HF_CACHE_DIR}"
  "$PYTHON_BIN" SchrodingerBridge/tools/experiments/build_offline_dino_pairing_cache.py \
    "${dino_cache_args[@]}"
else
  echo "[620] DINO cache exists: ${DINO_CACHE}"
fi

if [ ! -f "$PAIRING_PLAN" ]; then
  echo "[620] building DINO pairing plan: ${PAIRING_PLAN}"
  mkdir -p "$(dirname "$PAIRING_PLAN")"
  "$PYTHON_BIN" SchrodingerBridge/tools/experiments/build_offline_dino_pairing_plan.py \
    --cache "$DINO_CACHE" \
    --output "$PAIRING_PLAN" \
    --topk 8 \
    --styles "$STYLES"
else
  echo "[620] DINO pairing plan exists: ${PAIRING_PLAN}"
fi

"$PYTHON_BIN" SchrodingerBridge/tools/probe_620_path_liveness.py --device cpu

"$PYTHON_BIN" - <<PY
import json
import os
from pathlib import Path

base = Path(os.environ["REMOTE_ROOT"]) / os.environ["CONFIG_REL"]
payload = json.loads(base.read_text(encoding="utf-8"))
batch_size = int(os.environ["BATCH_SIZE"])
full_eval_batch_size = int(os.environ["FULL_EVAL_BATCH_SIZE"])
full_eval_vae_decode_batch_size = int(os.environ["FULL_EVAL_VAE_DECODE_BATCH_SIZE"])
if batch_size % 16 != 0:
    raise SystemExit(f"BATCH_SIZE must be divisible by 16, got {batch_size}")
if full_eval_batch_size % 16 != 0:
    raise SystemExit(f"FULL_EVAL_BATCH_SIZE must be divisible by 16, got {full_eval_batch_size}")
if full_eval_vae_decode_batch_size % 16 != 0:
    raise SystemExit(
        f"FULL_EVAL_VAE_DECODE_BATCH_SIZE must be divisible by 16, got {full_eval_vae_decode_batch_size}"
    )
training = payload.setdefault("training", {})
training["num_epochs"] = int(os.environ["EPOCHS"])
training["batch_size"] = batch_size
training["pin_memory"] = False
training["prefetch_factor"] = 1
training["full_eval_batch_size"] = full_eval_batch_size
training["full_eval_vae_decode_batch_size"] = full_eval_vae_decode_batch_size
full_eval = payload.setdefault("full_eval", {})
full_eval["batch_size"] = full_eval_batch_size
full_eval["vae_decode_batch_size"] = full_eval_vae_decode_batch_size
save_dir = f"./exp/620_spatial_bridge/{os.environ['RUN_NAME']}"
if os.environ["STAGE"] == "smoke":
    save_dir += "_smoke"
payload.setdefault("checkpoint", {})["save_dir"] = save_dir
payload.setdefault("ablation", {})["name"] = os.environ["RUN_NAME"]
payload.setdefault("ablation", {})["stage"] = os.environ["STAGE"]
out = Path(os.environ["REMOTE_SB"]) / "configs" / "_generated_620_launch.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("[620] launch_config", out)
print("[620] save_dir", save_dir)
PY

"$PYTHON_BIN" SchrodingerBridge/src/run.py --config SchrodingerBridge/configs/_generated_620_launch.json
