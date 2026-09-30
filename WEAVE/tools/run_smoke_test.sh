#!/bin/bash
# Run 620_nswd_gate03_smoke (NSWD sigma=0.02 + gate=0.3 + larger head)
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge/src:$PYTHONPATH

python3 -u src/train.py \
    --config /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/config.json \
    2>&1 | tee /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/train.log