#!/usr/bin/env bash
set -uo pipefail
cp /mnt/c/Users/Administrator/train_pixel256_fg.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_pixel256_fg.sh
echo TRAIN_SCRIPT_COPIED
grep -E "timeout|batch" /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_pixel256_fg.sh
