#!/usr/bin/env bash
set -euo pipefail

# Phase 4 620 experiment launcher - runs on remote WSL (3060, 12G VRAM)
# Usage: bash run_phase4_plan.sh [A|B|C|D|E|F|all] [smoke|formal]
# Each phase runs sequentially, smoke (6 epochs) first, then formal (10 epochs).

REMOTE_ROOT="/mnt/i/Github/Latent_Style"
SB_ROOT="${REMOTE_ROOT}/SchrodingerBridge"
PYTHON="/home/xy/venvs/samam312/bin/python"
LATENT_ROOT="/mnt/i/wikiart_distinct5_samam_512_latents_ema/train"
IMAGE_ROOT="/mnt/i/datasets/wikiart_distinct5_512_images/train"
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

# Cache paths
CACHE_DEFAULT="/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt"
CACHE_L8="/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_layer8_cache.pt"
CACHE_MULTI="/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_layers_4_8_11_cache.pt"

# Pairing plan paths
PLAN_DEFAULT="${LATENT_ROOT}/.latent_cache/dino_pairing_top8.pt"
PLAN_L8="${LATENT_ROOT}/.latent_cache/dino_pairing_layer8.pt"
PLAN_MULTI="${LATENT_ROOT}/.latent_cache/dino_pairing_layers_4_8_11.pt"
PLAN_COMPLEXITY="${LATENT_ROOT}/.latent_cache/dino_pairing_complexity.pt"

export PYTHONPATH="${SB_ROOT}/src:${SB_ROOT}/tools:${SB_ROOT}/tools/experiments"
export DINO_ALLOW_NETWORK=0

PHASE="${1:-all}"
MODE="${2:-smoke}"

if [ "$MODE" = "smoke" ]; then
  EPOCHS=6
elif [ "$MODE" = "formal" ]; then
  EPOCHS=10
else
  EPOCHS="$MODE"
fi

BATCH=64
FULL_EVAL_BATCH=8
FULL_EVAL_VAE_BATCH=8

log() { echo "[$(date +%H:%M:%S)] [P4] $*"; }

run_config() {
  local run_name="$1"
  local overrides="$2"
  local config_path="${SB_ROOT}/configs/620_spatial_bridge_base.json"

  if [ ! -f "$config_path" ]; then
    log "ERROR config not found: $config_path"
    return 1
  fi

  log "=== START ${run_name} (${MODE}, ${EPOCHS} epochs) ==="

  # Generate launch config dynamically using Python
  "$PYTHON" - <<PY
import json
from pathlib import Path

# Load baseline config
base = Path("${config_path}")
payload = json.loads(base.read_text(encoding="utf-8"))

# Override training epochs and batch size
training = payload.setdefault("training", {})
training["num_epochs"] = int("${EPOCHS}")
training["batch_size"] = int("${BATCH}")
training["pin_memory"] = False
training["prefetch_factor"] = 1
training["full_eval_batch_size"] = int("${FULL_EVAL_BATCH}")
training["full_eval_vae_decode_batch_size"] = int("${FULL_EVAL_VAE_BATCH}")
training["full_eval_defer_until_training_end"] = False
training["full_eval_each_epoch"] = True
training["full_eval_force_regen"] = True

full_eval = payload.setdefault("full_eval", {})
full_eval["batch_size"] = int("${FULL_EVAL_BATCH}")
full_eval["vae_decode_batch_size"] = int("${FULL_EVAL_VAE_BATCH}")

# Set save dir and ablation details
save_dir = f"./exp/620_spatial_bridge/${run_name}"
if "${MODE}" == "smoke":
    save_dir += "_smoke"
payload.setdefault("checkpoint", {})["save_dir"] = save_dir
payload.setdefault("ablation", {})["name"] = "${run_name}"
payload.setdefault("ablation", {})["stage"] = "${MODE}"

# Apply specific experiment overrides
overrides = json.loads('${overrides}')
for section, kv in overrides.items():
    s_dict = payload.setdefault(section, {})
    for k, v in kv.items():
        s_dict[k] = v

out = Path("${SB_ROOT}/configs/_generated_620_launch.json")
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[P4] save_dir={save_dir}")
PY

  # Run training
  "$PYTHON" "${SB_ROOT}/src/run.py" --config "${SB_ROOT}/configs/_generated_620_launch.json"
  local rc=$?
  if [ $rc -eq 0 ]; then
    log "=== DONE ${run_name} ==="
  else
    log "=== FAILED ${run_name} (rc=$rc) ==="
  fi
  return $rc
}

ensure_caches() {
  log "Ensuring DINO caches..."
  
  # 1. Default cache (if missing)
  if [ ! -f "$CACHE_DEFAULT" ]; then
    log "Building default DINO cache..."
    mkdir -p "$(dirname "$CACHE_DEFAULT")"
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_cache.py" \
      --latent-root "$LATENT_ROOT" --image-root "$IMAGE_ROOT" --output "$CACHE_DEFAULT"
  fi
  if [ ! -f "$PLAN_DEFAULT" ]; then
    log "Building default pairing plan..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_plan.py" \
      --cache "$CACHE_DEFAULT" --output "$PLAN_DEFAULT" --topk 8 --styles "$STYLES"
  fi

  # 2. Layer 8 cache (for Block A1 and A3)
  if [ ! -f "$CACHE_L8" ]; then
    log "Building Layer 8 DINO cache..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_cache.py" \
      --latent-root "$LATENT_ROOT" --image-root "$IMAGE_ROOT" --output "$CACHE_L8" --layers 8
  fi
  if [ ! -f "$PLAN_L8" ]; then
    log "Building Layer 8 pairing plan..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_plan.py" \
      --cache "$CACHE_L8" --output "$PLAN_L8" --topk 8 --styles "$STYLES"
  fi

  # 3. Concatenated layers 4,8,11 cache (for Block A2)
  if [ ! -f "$CACHE_MULTI" ]; then
    log "Building layers [4,8,11] concat DINO cache..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_cache.py" \
      --latent-root "$LATENT_ROOT" --image-root "$IMAGE_ROOT" --output "$CACHE_MULTI" --layers 4,8,11
  fi
  if [ ! -f "$PLAN_MULTI" ]; then
    log "Building layers [4,8,11] pairing plan..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_plan.py" \
      --cache "$CACHE_MULTI" --output "$PLAN_MULTI" --topk 8 --styles "$STYLES"
  fi

  # 4. Complexity matching pairing plan (for Block F3)
  if [ ! -f "$PLAN_COMPLEXITY" ]; then
    log "Building complexity matching pairing plan..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_plan.py" \
      --cache "$CACHE_DEFAULT" --output "$PLAN_COMPLEXITY" --topk 8 --styles "$STYLES" \
      --complexity-matching --complexity-weight 0.2
  fi
}

# Block A: Style Encoder (3 experiments)
block_a() {
  log "========== Block A: Style Encoder =========="
  
  # A1: DINO layer 8 (baseline)
  run_config "620_block_a1_layer8" '{"model": {"tokenizer_dino_dim": 384}, "data": {"dino_cache_path": "'"$CACHE_L8"'", "pairing_cache_path": "'"$PLAN_L8"'"}}'

  # A2: DINO layers [4,8,11] concat
  run_config "620_block_a2_concat" '{"model": {"tokenizer_dino_dim": 1152}, "data": {"dino_cache_path": "'"$CACHE_MULTI"'", "pairing_cache_path": "'"$PLAN_MULTI"'"}}'

  # A3: DINO layer 8 + Trainable LocalCNN
  run_config "620_block_a3_local_cnn" '{"model": {"tokenizer_dino_dim": 384, "style_local_cnn_enabled": true}, "data": {"dino_cache_path": "'"$CACHE_L8"'", "pairing_cache_path": "'"$PLAN_L8"'"}}'
}

# Block B: Per-Region SWD (3 experiments)
block_b() {
  log "========== Block B: Per-Region SWD =========="
  
  # B1: Global SWD (baseline)
  run_config "620_block_b1_global_swd" '{"bridge": {"swd_scale_mode": "global"}}'

  # B2: 2-scale SWD
  run_config "620_block_b2_2scale_swd" '{"bridge": {"swd_scale_mode": "2-scale"}}'

  # B3: 3-scale SWD
  run_config "620_block_b3_3scale_swd" '{"bridge": {"swd_scale_mode": "3-scale"}}'

  # B4: Attention-weighted SWD
  run_config "620_block_b4_weighted_swd" '{"bridge": {"swd_scale_mode": "attention-weighted"}}'
}

# Block C: Skip Connection Ratio (4 experiments)
block_c() {
  log "========== Block C: Skip Connections =========="

  # C1: alpha=1.0 (baseline)
  run_config "620_block_c1_alpha1p0" '{"model": {"style_shortcut_alpha": 1.0}}'

  # C2: alpha=[1.0, 0.7, 0.5, 0.3] per-layer
  run_config "620_block_c2_alpha_decay" '{"model": {"style_shortcut_alpha": [1.0, 0.7, 0.5, 0.3]}}'

  # C3: alpha=0.5 all layers
  run_config "620_block_c3_alpha0p5" '{"model": {"style_shortcut_alpha": 0.5}}'

  # C4: Learnable gating
  run_config "620_block_c4_alpha_learnable" '{"model": {"style_shortcut_alpha": "learnable"}}'
}

# Block D: Cross-Attention Query Source (4 experiments)
block_d() {
  log "========== Block D: Cross-Attention Query =========="

  # D1: Q = concat(skip, bottleneck) (baseline)
  run_config "620_block_d1_q_concat" '{"model": {"style_query_source": "concat"}}'

  # D2: Q = bottleneck only
  run_config "620_block_d2_q_sa_only" '{"model": {"style_query_source": "sa_out_only"}}'

  # D3: Q = content DINO patches
  run_config "620_block_d3_q_dino" '{"model": {"style_query_source": "content_dino"}}'

  # D4: Skip coarse layers Cross-Attention
  run_config "620_block_d4_skip_coarse" '{"model": {"style_cross_attn_skip_coarse": true}}'
}

# Block E: Attention Sparsification (3 experiments)
block_e() {
  log "========== Block E: Attention Sparsification =========="

  # E1: Softmax attention (baseline)
  run_config "620_block_e1_softmax" '{"model": {"style_attn_topk": 0}}'

  # E2: Top-k attention (k=16)
  run_config "620_block_e2_topk16" '{"model": {"style_attn_topk": 16}}'

  # E3: Attention entropy regularization (lambda=0.01)
  run_config "620_block_e3_entropy_reg" '{"bridge": {"w_attn_entropy_reg": 0.01}}'
}

# Block F: OT Pairing (3 experiments)
block_f() {
  log "========== Block F: OT Pairing =========="

  # F1: DINO top-1 fixed (baseline)
  run_config "620_block_f1_top1_fixed" '{"data": {"pairing_cache_active_topk": 1, "pairing_cache_sample_mode": "top1"}}'

  # F2: DINO top-5 rotation
  run_config "620_block_f2_top5_uniform" '{"data": {"pairing_cache_active_topk": 5, "pairing_cache_topk": 5, "pairing_cache_sample_mode": "uniform"}}'

  # F3: Attention complexity matching
  run_config "620_block_f3_complexity_matching" '{"data": {"pairing_cache_path": "'"$PLAN_COMPLEXITY"'"}}'
}

# Main
cd "$REMOTE_ROOT"
ensure_caches

case "$PHASE" in
  A) block_a ;;
  B) block_b ;;
  C) block_c ;;
  D) block_d ;;
  E) block_e ;;
  F) block_f ;;
  all)
    block_a
    block_b
    block_c
    block_d
    block_e
    block_f
    ;;
  *)
    echo "Usage: $0 [A|B|C|D|E|F|all] [smoke|formal]"
    exit 1
    ;;
esac

log "========== Phase ${PHASE} (${MODE}) complete =========="
