#!/bin/bash
echo "=== Process check ==="
ps -p 108226 -o pid,etime,cmd 2>/dev/null || echo "PID 108226 not found"
echo ""
echo "=== Recent log ==="
tail -40 /mnt/i/exp_samam_latent_train.log 2>/dev/null || echo "no log yet"
echo ""
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
echo ""
echo "=== Output dir ==="
ls -la /mnt/i/exp_samam_latent/ 2>/dev/null | tail -20
