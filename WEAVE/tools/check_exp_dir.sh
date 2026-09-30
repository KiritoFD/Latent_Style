#!/bin/bash
for exp in 620_intrinsic_v2 620_lowswd_formal 620_film_formal; do
    echo "--- $exp ---"
    ls /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/$exp/ 2>/dev/null | head -20
done
