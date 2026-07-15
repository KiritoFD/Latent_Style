#!/bin/bash
set -euo pipefail
cd /mnt/i/Github/Latent_Style/SchrodingerBridge/src
export PYTHONPATH="/mnt/i/Github/Latent_Style/SchrodingerBridge/src:${PYTHONPATH:-}"

echo "Launching 620_nswd_gate03_smoke (NSWD sigma=0.02 + gate=0.3 + larger head)"
nohup python3 run.py \
    --config /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/config.json \
    > /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/train.log 2>&1 &
PID=$!
echo "PID: $PID"
echo "Monitor: tail -f /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/train.log"