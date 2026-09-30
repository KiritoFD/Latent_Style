#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
OT_FIXED_BATCH_SIZE="${OT_FIXED_BATCH_SIZE:-16}"
PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE="${PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE:-20}"
STYLE_SWEEP_FIXED_BATCH_SIZE="${STYLE_SWEEP_FIXED_BATCH_SIZE:-20}"

echo "============================================"
echo "  $(date) START phase618 pipeline"
echo "  phase=old_ot_rerun -> plain_path_distill -> style_sweep"
echo "  ot_fixed_batch_size=$OT_FIXED_BATCH_SIZE"
echo "  plain_path_distill_fixed_batch_size=$PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE"
echo "  style_sweep_fixed_batch_size=$STYLE_SWEEP_FIXED_BATCH_SIZE"
echo "============================================"

cd "$ROOT"

OT_FIXED_BATCH_SIZE="$OT_FIXED_BATCH_SIZE" bash tools/experiments/run_phase618_ot_rerun.sh
PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE="$PLAIN_PATH_DISTILL_FIXED_BATCH_SIZE" bash tools/experiments/run_phase618_plain_path_distill.sh
STYLE_SWEEP_FIXED_BATCH_SIZE="$STYLE_SWEEP_FIXED_BATCH_SIZE" bash tools/experiments/run_phase618_style_sweep.sh

echo "============================================"
echo "  $(date) DONE phase618 pipeline"
echo "============================================"
