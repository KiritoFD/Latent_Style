#!/bin/bash
echo "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
echo ""
echo "=== All python processes ==="
ps -ef | grep -E "python|train_SaMam|gen_samst" | grep -v grep
echo ""
echo "=== Kill any leftover gen_samst processes ==="
pkill -f "gen_samst_latent" 2>/dev/null && echo "killed gen_samst" || echo "no gen_samst running"
echo ""
echo "=== SaMam training log ==="
tail -30 /mnt/i/exp_samam_latent_train.log 2>/dev/null
echo ""
echo "=== SaMam training process ==="
ps -p 108395 -o pid,etime,cmd 2>/dev/null || echo "PID 108395 not found"
