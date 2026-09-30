#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/i/Github/Latent_Style}"
PYTHON_BIN="${PYTHON_BIN:-/home/xy/venvs/samam312/bin/python}"
SB_ROOT="$ROOT/SchrodingerBridge"
CHECKPOINT="${CHECKPOINT:-$ROOT/exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt}"
TEST_DIR="${TEST_DIR:-/mnt/i/wikiart_distinct5_samam_512_classview/test}"
STYLE_BANK_ROOT="${STYLE_BANK_ROOT:-$TEST_DIR}"
CACHE_DIR="${CACHE_DIR:-$ROOT/eval_cache}"
CLIP_HF_CACHE_DIR="${CLIP_HF_CACHE_DIR:-$ROOT/eval_cache/hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/inmortal-exp/phase2_style_covariant_lowanchor050e9}"
LOG_ROOT="${LOG_ROOT:-$SB_ROOT/docs/experiments/phase2_fiber_bundle/616/logs/style_covariant_probe}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

CONFIGS=(
  phase2_eval_i2sb_noise_sigma0p0_lowanchor050e9
  phase2_eval_i2sb_noise_gaussian_sigma0p5_lowanchor050e9
  phase2_eval_i2sb_noise_stylecov_sigma0p5_lowanchor050e9
  phase2_eval_i2sb_noise_gaussian_sigma0p8_lowanchor050e9
  phase2_eval_i2sb_noise_stylecov_sigma0p8_lowanchor050e9
  phase2_eval_i2sb_noise_gaussian_sigma1p2_lowanchor050e9
  phase2_eval_i2sb_noise_stylecov_sigma1p2_lowanchor050e9
)

echo "[style_covariant_probe] root=$ROOT"
echo "[style_covariant_probe] checkpoint=$CHECKPOINT"
echo "[style_covariant_probe] output_root=$OUTPUT_ROOT"
echo "[style_covariant_probe] log_root=$LOG_ROOT"

for stem in "${CONFIGS[@]}"; do
  override="$SB_ROOT/configs/aaai2027/${stem}.json"
  out_dir="$OUTPUT_ROOT/$stem/epoch_0009"
  log_path="$LOG_ROOT/${stem}.log"
  echo "[style_covariant_probe] start $stem"
  {
    echo "=== START $(date --iso-8601=seconds) ==="
    echo "CONFIG: $override"
    echo "OUTPUT: $out_dir"
    "$PYTHON_BIN" "$SB_ROOT/tools/experiments/run_phase2_eval_only_override.py" \
      --checkpoint "$CHECKPOINT" \
      --config-override "$override" \
      --output "$out_dir" \
      --test-dir "$TEST_DIR" \
      --cache-dir "$CACHE_DIR" \
      --clip-hf-cache-dir "$CLIP_HF_CACHE_DIR" \
      --seed "$SEED"
    if [[ "$stem" == *"stylecov"* ]]; then
      "$PYTHON_BIN" - <<'PY' "$out_dir/summary.json"
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
settings = summary.get("settings", {})
runtime = settings.get("lgt_runtime_observability", {}) or {}
subset = {
    k: runtime.get(k)
    for k in [
        "style_noise_family_style_covariant",
        "style_noise_family_gaussian",
        "style_noise_bank_active",
        "style_noise_amp_mean",
        "style_noise_amp_std",
        "style_noise_post_std",
        "style_noise_amplitude_power",
        "style_noise_fallback_gaussian",
    ]
}
print("[style_covariant_probe] runtime_subset=" + json.dumps(subset, ensure_ascii=True))
PY
    fi
    echo "=== END $(date --iso-8601=seconds) ==="
  } 2>&1 | tee "$log_path"
done
