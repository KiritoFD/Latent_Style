#!/usr/bin/env bash
set -uo pipefail
cp /mnt/c/Users/Administrator/630_pixel_256_photo2art.json /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/
cp /mnt/c/Users/Administrator/eval_pixel256_epoch10.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
cp /mnt/c/Users/Administrator/methods_ours_pixel256.json /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/eval_pixel256_epoch10.sh
echo COPIED
grep -E "batch_size|save_dir|style_attn_mode" /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_pixel_256_photo2art.json
echo "===GPU BEFORE==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
