#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="exp/aaai2027_phase616_clean_unbalanced_dummy_vertical_affine_dummy055_tau025_step600_eval_remote/epoch_0001.pt"
BASE_SUMMARY="exp/aaai2027_phase616_clean_unbalanced_dummy_vertical_affine_dummy055_tau025_step600_eval_remote/full_eval_transfer_step600/epoch_0001/summary.json"
RUN_ROOT="exp/aaai2027_phase616_clean_unbalanced_dummy_vertical_affine_dummy055_tau025_step600_eval_remote/eval_frontier"
LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_eval_frontier"
RESULTS_TSV="${LOG_DIR}/eval_frontier.tsv"
TEST_DIR="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
CLIP_HF_CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache/hf"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

printf "variant\tclip_style\tcontent_lpips\twall_total_sec\tgeneration_sec\tvae_decode_sec\tsummary_path\n" > "${RESULTS_TSV}"

parse_summary_fields() {
  local summary_path="$1"
  python3 - "${summary_path}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
metrics = ((summary.get("analysis") or {}).get("style_transfer_ability") or {})
timings = summary.get("timings_sec") or {}
print(
    metrics.get("clip_style"),
    metrics.get("content_lpips"),
    timings.get("wall_total"),
    timings.get("lancet_generation"),
    timings.get("vae_decode"),
)
PY
}

record_summary() {
  local variant="$1"
  local summary_path="$2"
  local clip_style content_lpips wall_total generation_sec vae_decode_sec
  read -r clip_style content_lpips wall_total generation_sec vae_decode_sec < <(parse_summary_fields "${summary_path}")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" \
    "${clip_style}" \
    "${content_lpips}" \
    "${wall_total}" \
    "${generation_sec}" \
    "${vae_decode_sec}" \
    "${summary_path}" >> "${RESULTS_TSV}"
  echo "[phase616_eval_frontier] ${variant} clip_style=${clip_style} content_lpips=${content_lpips} wall_total_sec=${wall_total} generation_sec=${generation_sec} vae_decode_sec=${vae_decode_sec}" >&2
  printf "%s %s %s %s %s\n" "${clip_style}" "${content_lpips}" "${wall_total}" "${generation_sec}" "${vae_decode_sec}"
}

run_override() {
  local override_rel="$1"
  local override_stem
  local output_dir
  local summary_path

  override_stem="$(basename "${override_rel}" .json)"
  output_dir="${RUN_ROOT}/${override_stem}"
  summary_path="${output_dir}/summary.json"

  echo "[phase616_eval_frontier] running ${override_stem}" >&2
  "${PYTHON_BIN}" tools/experiments/run_phase2_eval_only_override.py \
    --checkpoint "${CHECKPOINT}" \
    --config-override "${override_rel}" \
    --output "${output_dir}" \
    --test-dir "${TEST_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --clip-hf-cache-dir "${CLIP_HF_CACHE_DIR}" \
    --device cuda \
    --force-regen >&2
  record_summary "${override_stem}" "${summary_path}"
}

geq_threshold() {
  local value="$1"
  local threshold="$2"
  python3 - "${value}" "${threshold}" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
}

read -r base_clip base_lpips base_wall base_gen base_decode < <(record_summary "baseline_step600" "${BASE_SUMMARY}")
read -r s110_clip s110_lpips s110_wall s110_gen s110_decode < <(run_override "configs/aaai2027/phase616_eval_style_overdrive_s110_dummy055_tau025_step600.json")

if geq_threshold "${s110_clip}" "0.74"; then
  echo "[phase616_eval_frontier] s110 already crossed clip_style>=0.74; stopping early." >&2
  cat "${RESULTS_TSV}"
  exit 0
fi

next_override="configs/aaai2027/phase616_eval_style_overdrive_s135_dummy055_tau025_step600.json"
next_label="s135"
if geq_threshold "${s110_clip}" "0.72"; then
  next_override="configs/aaai2027/phase616_eval_style_overdrive_s120_dummy055_tau025_step600.json"
  next_label="s120"
fi

echo "[phase616_eval_frontier] branching to ${next_label}" >&2
run_override "${next_override}" >/dev/null

cat "${RESULTS_TSV}"
