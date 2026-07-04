#!/bin/bash
echo "=== Process check (new PID 108395) ==="
ps -p 108395 -o pid,etime,cmd 2>/dev/null || echo "PID 108395 not found"
echo ""
echo "=== All train_SaMam processes ==="
ps -ef | grep train_SaMam | grep -v grep
echo ""
echo "=== Recent log ==="
tail -50 /mnt/i/exp_samam_latent_train.log 2>/dev/null || echo "no log yet"
echo ""
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
echo ""
echo "=== Output dir ==="
ls -la /mnt/i/exp_samam_latent/ 2>/dev/null | tail -20
