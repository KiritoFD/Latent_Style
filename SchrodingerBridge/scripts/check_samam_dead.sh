#!/bin/bash
echo "=== Process check ==="
ps -p 108395 -o pid,etime,cmd 2>/dev/null || echo "PID 108395 not found"
echo ""
echo "=== All python procs ==="
ps -ef | grep python | grep -v grep | head -10
echo ""
echo "=== Full log ==="
cat /mnt/i/exp_samam_latent_train.log 2>/dev/null
echo ""
echo "=== Output dir ==="
ls -la /mnt/i/exp_samam_latent/ 2>/dev/null
