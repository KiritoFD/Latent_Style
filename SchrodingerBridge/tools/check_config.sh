#!/bin/bash
echo "=== Check experiment configs ==="
for exp in 620_intrinsic_v2 620_lowswd_formal 620_film_formal; do
    echo "--- $exp ---"
    cat /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/$exp/config.yaml 2>/dev/null | grep -i "save_gen\|save_image\|full_eval" || echo "(no config.yaml or no match)"
    cat /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/$exp/train_config.yaml 2>/dev/null | grep -i "save_gen\|save_image\|full_eval" || echo "(no train_config.yaml or no match)"
    ls /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/$exp/*.yaml 2>/dev/null
done
