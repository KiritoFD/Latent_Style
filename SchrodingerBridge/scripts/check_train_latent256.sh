#!/bin/bash
echo "===PROCESS==="
ps -ef | grep -E "run\.py|samam312" | grep -v grep
echo "===LOG (last 80 lines)==="
tail -80 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log 2>/dev/null || echo "LOG NOT FOUND"
echo "===LOG SIZE==="
ls -la /mnt/i/exp_256_photo2art/ 2>/dev/null
echo "===NVIDIA==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
echo "===CKPT DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/ 2>/dev/null || echo "NO CKPT DIR YET"
