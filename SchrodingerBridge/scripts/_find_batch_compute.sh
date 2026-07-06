#!/usr/bin/env bash
echo "===find batch_compute script==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts -name "batch_compute*" 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge -name "batch_compute_photo2art*" 2>/dev/null
echo "===check pixel256 images dir==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003/images/ 2>/dev/null | head -10
echo "===count pixel256 images==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003/images -name "*.png" 2>/dev/null | wc -l
echo "===check how baselines were computed==="
head -30 /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/batch_compute_photo2art.py 2>/dev/null
