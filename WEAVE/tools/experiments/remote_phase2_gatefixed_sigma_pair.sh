#!/usr/bin/env bash
set -euo pipefail

SIGMA_TAG="${1:-0p01}"
ROOT="${LATENT_STYLE_ROOT:-/mnt/i/Github/Latent_Style}"
REPO="$ROOT/SchrodingerBridge"
PY="${PYTHON_BIN:-python3}"
CKPT="${CKPT:-$ROOT/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt}"
TEST_DIR="${TEST_DIR:-/mnt/i/wikiart_distinct5_samam_512_classview/test}"
CACHE_DIR="${CACHE_DIR:-$ROOT/eval_cache}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/eval_cache/hf}"
OUT_ROOT="${OUT_ROOT:-exp/inmortal-exp/phase2_fiber_sde_gatefixed_k070_e3}"

cd "$REPO"

run_one() {
  local mode="$1"
  local config="$REPO/configs/aaai2027/phase2_fiber_sde_${mode}_sigma${SIGMA_TAG}.json"
  local output="$OUT_ROOT/phase2_fiber_sde_${mode}_sigma${SIGMA_TAG}/epoch_0003"
  "$PY" tools/experiments/run_phase2_eval_only_override.py \
    --checkpoint "$CKPT" \
    --config-override "$config" \
    --output "$output" \
    --test-dir "$TEST_DIR" \
    --cache-dir "$CACHE_DIR" \
    --clip-hf-cache-dir "$HF_CACHE_DIR" \
    --device cuda \
    --force-regen
}

run_one iso
run_one fiber
