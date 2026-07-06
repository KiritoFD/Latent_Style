#!/usr/bin/env bash
set -uo pipefail
echo "===PIXEL CACHE==="
ls -la /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/packed/ 2>/dev/null | head -10
echo "===CONFIG==="
grep -E "latent_cache_dir|data_root|batch_size|num_epochs|use_gradient_checkpointing|save_dir" /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_pixel_256_photo2art.json
echo "===GPU==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "===EXISTING CKPT DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/ 2>/dev/null || echo "NOT EXISTS"
echo "===DISK FREE==="
df -h /mnt/i | tail -2
