#!/bin/bash
set -e
mkdir -p /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/other5_eval/samam
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python3 scripts/samam_latent/gen_samam_latent.py \
  --checkpoint /mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/step-step=020000.ckpt \
  --test-root /mnt/i/Github/Latent_Style/Dataset/other5_512/test \
  --output-root /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/other5_eval/samam \
  --style-names Abstract_Expressionism,Art_Nouveau_Modern,Cubism,Expressionism,Symbolism \
  --num-src 30 \
  2>&1 | tee /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/other5_eval/samam_log.txt
