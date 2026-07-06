#!/usr/bin/env bash
set -uo pipefail
echo "===PIXEL256 CKPT DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/ 2>/dev/null
echo "===PIXEL256 LOG TAIL==="
tail -30 /mnt/i/exp_256_photo2art/_train_pixel256_photo2art.log 2>/dev/null
echo "===GPU NOW==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo "===DISK FREE==="
df -h /mnt/i | tail -2
echo "===EXP_ABLATION_620 DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/exp_ablation_620/ 2>/dev/null | head -30
echo "===FIND ALL ABLATION DIRS==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -maxdepth 2 -type d -name "*ablation*" 2>/dev/null | head -20
