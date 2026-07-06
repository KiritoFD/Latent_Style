#!/bin/bash
# Copy config to correct location and launch training
cp /mnt/c/Users/Administrator/630_latent_256_photo2art.json /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_latent_256_photo2art.json
echo "[INFO] Config copied."
# Verify the eval settings
grep -E "full_eval_each_epoch|full_eval_defer" /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_latent_256_photo2art.json
echo "---"
# Launch training
bash /mnt/c/Users/Administrator/launch_train_latent256_setsid.sh
