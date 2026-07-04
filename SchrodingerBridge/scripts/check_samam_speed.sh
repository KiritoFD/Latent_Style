#!/bin/bash
echo "=== Wait 60s and check iter speed ==="
sleep 60
echo "=== Recent log ==="
tail -10 /mnt/i/exp_samam_latent_train.log 2>/dev/null
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
