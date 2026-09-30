#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="aaai2027_phase616_clean_unbalanced_dummy_vertical_affine_dummy055_tau025_step600_eval_remote"
RUN_ROOT="exp/${RUN_NAME}"
LOG_DIR="docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_rebuild_step_frontier"
RESULTS_TSV="${LOG_DIR}/step_frontier.tsv"
OVERRIDE="configs/aaai2027/phase616_eval_baseline_dummy055_tau025_stepfrontier.json"
TEST_DIR="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
CLIP_HF_CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache/hf"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/step_frontier"

printf "checkpoint\tclip_style\tcontent_lpips\twall_total_sec\tgeneration_sec\tvae_decode_sec\tsummary_path\n" > "${RESULTS_TSV}"

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
  local label="$1"
  local summary_path="$2"
  local clip_style content_lpips wall_total generation_sec vae_decode_sec
  read -r clip_style content_lpips wall_total generation_sec vae_decode_sec < <(parse_summary_fields "${summary_path}")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" \
    "${clip_style}" \
    "${content_lpips}" \
    "${wall_total}" \
    "${generation_sec}" \
    "${vae_decode_sec}" \
    "${summary_path}" >> "${RESULTS_TSV}"
  echo "[phase616_step_frontier] ${label} clip_style=${clip_style} content_lpips=${content_lpips} wall_total_sec=${wall_total}" >&2
}

run_checkpoint_eval() {
  local checkpoint_rel="$1"
  local label="$2"
  local output_dir="${RUN_ROOT}/step_frontier/${label}"
  local summary_path="${output_dir}/summary.json"

  echo "[phase616_step_frontier] running ${label} from ${checkpoint_rel}" >&2
  "${PYTHON_BIN}" tools/experiments/run_phase2_eval_only_override.py \
    --checkpoint "${checkpoint_rel}" \
    --config-override "${OVERRIDE}" \
    --output "${output_dir}" \
    --test-dir "${TEST_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --clip-hf-cache-dir "${CLIP_HF_CACHE_DIR}" \
    --device cuda \
    --force-regen >/dev/null
  record_summary "${label}" "${summary_path}"
}

run_checkpoint_eval "${RUN_ROOT}/step_000200.pt" "step_0200"
run_checkpoint_eval "${RUN_ROOT}/step_000400.pt" "step_0400"
run_checkpoint_eval "${RUN_ROOT}/step_000600.pt" "step_0600"

cat "${RESULTS_TSV}"
