#!/bin/bash
# 628 inference ablation runner - single experiment
# Usage: bash 628_run_one_infer.sh <exp_name> <overrides_json>
# Example: bash 628_run_one_infer.sh I9_fiber_only '{"fiber_only_endpoint": true}'

set -e
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
PYTHON=/home/xy/venvs/samam312/bin/python3

EXP_NAME=$1
OVERRIDES=$2

echo "[628] Running $EXP_NAME with overrides: $OVERRIDES"
echo "[628] Start: $(date)"

$PYTHON 628_infer_ablation.py "$EXP_NAME" "$OVERRIDES"

echo "[628] Done: $(date)"
