#!/usr/bin/env bash
echo "===exp_baselines structure==="
ls /mnt/i/Github/Latent_Style/exp_baselines/ 2>/dev/null | head -20
echo "===256 photo2art results==="
find /mnt/i/Github/Latent_Style -path "*256*photo2art*" -name "summary.json" 2>/dev/null
find /mnt/i/Github/Latent_Style -path "*photo2art*256*" -name "summary.json" 2>/dev/null
echo "===exp_256_photo2art dir==="
ls /mnt/i/exp_256_photo2art/ 2>/dev/null
find /mnt/i/exp_256_photo2art -name "summary.json" 2>/dev/null | head -20
echo "===adain/wct results==="
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*adain*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*wct*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*seedream*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*identity*" 2>/dev/null | head -5
