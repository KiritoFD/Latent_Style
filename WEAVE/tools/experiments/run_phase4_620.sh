#!/usr/bin/env bash
set -euo pipefail

# Phase 4 620 experiment launcher - runs on remote 3060 WSL
# Usage: bash run_phase4_620.sh [A|B|C|D|E|all]
# Each phase runs sequentially, smoke (6 epochs) first, then formal for best.

REMOTE_ROOT="/mnt/f/Github/Latent_Style"
SB_ROOT="${REMOTE_ROOT}/SchrodingerBridge"
PYTHON="/home/xy/venvs/samam312/bin/python"
LATENT_ROOT="/mnt/f/wikiart_distinct5_samam_512_latents_ema/train"
IMAGE_ROOT="/mnt/f/wikiart_distinct5_512_images/train"
DINO_CACHE="/mnt/f/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt"
PAIRING_PLAN="${LATENT_ROOT}/.latent_cache/dino_pairing_top8.pt"
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

export PYTHONPATH="${SB_ROOT}/src:${SB_ROOT}/tools:${SB_ROOT}/tools/experiments"
export DINO_ALLOW_NETWORK=0

PHASE="${1:-all}"
EPOCHS_SMOKE=6
EPOCHS_FORMAL=10
BATCH=64
FULL_EVAL_BATCH=8
FULL_EVAL_VAE_BATCH=8

log() { echo "[$(date +%H:%M:%S)] [P4] $*"; }

run_config() {
  local config_name="$1"
  local run_name="$2"
  local epochs="$3"
  local stage="$4"
  local config_path="${SB_ROOT}/configs/${config_name}"

  if [ ! -f "$config_path" ]; then
    log "ERROR config not found: $config_path"
    return 1
  fi

  log "=== START ${run_name} (${stage}, ${epochs} epochs) ==="

  # Generate launch config
  "$PYTHON" - <<PY
import json, os
from pathlib import Path
base = Path("${config_path}")
payload = json.loads(base.read_text(encoding="utf-8"))
training = payload.setdefault("training", {})
training["num_epochs"] = int("${epochs}")
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
save_dir = f"./exp/620_spatial_bridge/${run_name}"
if "${stage}" == "smoke" and not save_dir.endswith("_smoke"):
    save_dir += "_smoke"
payload.setdefault("checkpoint", {})["save_dir"] = save_dir
payload.setdefault("ablation", {})["name"] = "${run_name}"
payload.setdefault("ablation", {})["stage"] = "${stage}"
out = Path("${SB_ROOT}/configs/_generated_620_launch.json")
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[P4] save_dir={save_dir}")
PY

  "$PYTHON" "${SB_ROOT}/src/run.py" --config "${SB_ROOT}/configs/_generated_620_launch.json"
  local rc=$?
  if [ $rc -eq 0 ]; then
    log "=== DONE ${run_name} ==="
  else
    log "=== FAILED ${run_name} (rc=$rc) ==="
  fi
  return $rc
}

# Build DINO cache if missing
ensure_dino_cache() {
  if [ ! -f "$DINO_CACHE" ]; then
    log "Building DINO cache..."
    mkdir -p "$(dirname "$DINO_CACHE")"
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_cache.py" \
      --latent-root "$LATENT_ROOT" --image-root "$IMAGE_ROOT" --output "$DINO_CACHE"
  else
    log "DINO cache exists"
  fi
  if [ ! -f "$PAIRING_PLAN" ]; then
    log "Building pairing plan..."
    "$PYTHON" "${SB_ROOT}/tools/experiments/build_offline_dino_pairing_plan.py" \
      --cache "$DINO_CACHE" --output "$PAIRING_PLAN" --topk 8 --styles "$STYLES"
  else
    log "Pairing plan exists"
  fi
}

# Phase A: Style Encoder (6 experiments)
phase_a() {
  log "========== Phase A: Style Encoder =========="

  # A1: topk scan (index param)
  run_config "620_spatial_bridge_topk4.json" "620_swd16_topk4_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_topk12.json" "620_swd16_topk12_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_topk16.json" "620_swd16_topk16_vlen004" "$EPOCHS_SMOKE" "smoke"

  # A2: DINO adapter enabled
  run_config "620_spatial_bridge_adapter.json" "620_adapter_swd16_vlen004" "$EPOCHS_SMOKE" "smoke"

  # A3: sample mode
  run_config "620_spatial_bridge_uniform.json" "620_swd16_uniform_vlen004" "$EPOCHS_SMOKE" "smoke"

  # A4: explore_prob
  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["data"]["pairing_cache_explore_prob"] = 0.2
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_explore02.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_explore02.json" "620_swd16_explore02_vlen004" "$EPOCHS_SMOKE" "smoke"
}

# Phase B: Cross-Attention / Model params (6 experiments)
phase_b() {
  log "========== Phase B: Cross-Attention / Model =========="

  # B1: gate init scan
  run_config "620_spatial_bridge_gate20.json" "620_swd16_gate020_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_gate25.json" "620_swd16_gate025_vlen004" "$EPOCHS_SMOKE" "smoke"

  # B2: MoE
  run_config "620_spatial_bridge_moe.json" "620_moe_swd16_vlen004" "$EPOCHS_SMOKE" "smoke"

  # B3: num_heads=8
  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["model"]["style_attn_num_heads"] = 8
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_heads8.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_heads8.json" "620_swd16_heads8_vlen004" "$EPOCHS_SMOKE" "smoke"

  # B4: num_res_blocks=6
  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["model"]["num_res_blocks"] = 6
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_blocks6.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_blocks6.json" "620_swd16_blocks6_vlen004" "$EPOCHS_SMOKE" "smoke"

  # B5: content-routed MoE
  run_config "620_spatial_bridge_contentkv.json" "620_contentkv_swd16_vlen004" "$EPOCHS_SMOKE" "smoke"
}

# Phase C: OT / Pairing (3 experiments)
phase_c() {
  log "========== Phase C: OT / Pairing =========="
  # Already covered in Phase A: topk scan, uniform, explore
  # Additional: rank_power scan
  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["data"]["pairing_cache_rank_power"] = 0.5
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_rankpow05.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_rankpow05.json" "620_swd16_rankpow05_vlen004" "$EPOCHS_SMOKE" "smoke"

  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["data"]["pairing_cache_rank_power"] = 2.0
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_rankpow20.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_rankpow20.json" "620_swd16_rankpow20_vlen004" "$EPOCHS_SMOKE" "smoke"

  # C3: dual_target_mix
  "$PYTHON" - <<PY
import json
from pathlib import Path
base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
p = json.loads(base.read_text())
p["bridge"]["single_step_swd_weight"] = 16.0
p["data"]["virtual_length_multiplier"] = 0.04
p["data"]["pairing_cache_dual_target_mix"] = 0.3
p["data"]["pairing_cache_dual_target_topk"] = 4
p["full_eval"] = p.get("full_eval", {})
p["full_eval"]["num_steps"] = 16
p["training"] = p.get("training", {})
p["training"]["batch_size"] = 64
out = Path("${SB_ROOT}/configs/620_spatial_bridge_dualmix03.json")
out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_dualmix03.json" "620_swd16_dualmix03_vlen004" "$EPOCHS_SMOKE" "smoke"
}

# Phase D: Loss / Supervisory signals (6 experiments)
phase_d() {
  log "========== Phase D: Loss params =========="

  # D1: sigma scan
  run_config "620_spatial_bridge_sigma001.json" "620_swd16_sigma001_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_sigma004.json" "620_swd16_sigma004_vlen004" "$EPOCHS_SMOKE" "smoke"

  # D2: edge weight scan
  run_config "620_spatial_bridge_edge005.json" "620_swd16_edge005_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_edge020.json" "620_swd16_edge020_vlen004" "$EPOCHS_SMOKE" "smoke"

  # D3: SWD=14 and SWD=18
  "$PYTHON" - <<PY
import json
from pathlib import Path
for swd in [14, 18]:
    base = Path("${SB_ROOT}/configs/620_spatial_bridge_base.json")
    p = json.loads(base.read_text())
    p["bridge"]["single_step_swd_weight"] = float(swd)
    p["data"]["virtual_length_multiplier"] = 0.04
    p["full_eval"] = p.get("full_eval", {})
    p["full_eval"]["num_steps"] = 16
    p["training"] = p.get("training", {})
    p["training"]["batch_size"] = 64
    out = Path(f"${SB_ROOT}/configs/620_spatial_bridge_swd{swd}.json")
    out.write_text(json.dumps(p, indent=2))
PY
  run_config "620_spatial_bridge_swd14.json" "620_swd14_vlen004" "$EPOCHS_SMOKE" "smoke"
  run_config "620_spatial_bridge_swd18.json" "620_swd18_vlen004" "$EPOCHS_SMOKE" "smoke"
}

# Phase E: Combination (will be run after analyzing A-D results)
phase_e() {
  log "========== Phase E: Combination =========="
  log "Phase E requires manual selection from A-D best results"
  log "Run: run_config <best_config> <run_name> \$EPOCHS_FORMAL formal"
}

# Main
cd "$REMOTE_ROOT"
ensure_dino_cache

case "$PHASE" in
  A) phase_a ;;
  B) phase_b ;;
  C) phase_c ;;
  D) phase_d ;;
  E) phase_e ;;
  all)
    phase_a
    phase_b
    phase_c
    phase_d
    phase_e
    ;;
  *)
    echo "Usage: $0 [A|B|C|D|E|all]"
    exit 1
    ;;
esac

log "========== Phase ${PHASE} complete =========="
