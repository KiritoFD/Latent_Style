#!/usr/bin/env bash
# Test VAE loading from local modelscope cache.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
LOG=/mnt/i/exp_256_photo2art/_vae_test.log
mkdir -p /mnt/i/exp_256_photo2art

"$PYTHON" /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/test_vae_load.py > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC"
echo "=== log ==="
tail -50 "$LOG"
