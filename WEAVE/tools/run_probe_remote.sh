#!/bin/bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
mkdir -p docs/620/fog/gradient_probe

python3 -u tools/probe_swd_gradient.py \
    --checkpoint /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/epoch_0008.pt \
    --config /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/config.json \
    --output-dir /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/gradient_probe \
    --device cuda \
    --batch-size 8 \
    2>&1 | tee /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/fog/gradient_probe/probe.log