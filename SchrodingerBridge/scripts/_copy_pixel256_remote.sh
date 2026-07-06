#!/usr/bin/env bash
set -uo pipefail
mkdir -p /mnt/i/exp_256_photo2art
cp /mnt/c/Users/Administrator/630_pixel_256_photo2art.json /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/
cp /mnt/c/Users/Administrator/train_pixel256_fg.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
cp /mnt/c/Users/Administrator/eval_pixel256_epoch10.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
cp /mnt/c/Users/Administrator/methods_ours_pixel256.json /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
cp /mnt/c/Users/Administrator/run_batch_compute_pixel256.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_pixel256_fg.sh
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/eval_pixel256_epoch10.sh
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/run_batch_compute_pixel256.sh
echo COPIED_OK
