#!/usr/bin/env bash
set -uo pipefail
SRC=/mnt/c/Users/Administrator/inference_pixel256_patch
DST=/mnt/i/Github/Latent_Style/SchrodingerBridge
cp "$SRC/inference.py" "$DST/src/utils/inference.py"
cp "$SRC/run_evaluation.py" "$DST/src/utils/run_evaluation.py"
cp /mnt/c/Users/Administrator/pixel256_eval_override.json "$DST/scripts/"
echo COPIED
ls -la "$DST/scripts/pixel256_eval_override.json" "$DST/src/utils/inference.py"
echo "===CHECK PASSTHROUGHVAE==="
grep -n "PassthroughVAE" "$DST/src/utils/inference.py" | head -5
grep -n "PassthroughVAE\|_is_pixel_space" "$DST/src/utils/run_evaluation.py" | head -10
