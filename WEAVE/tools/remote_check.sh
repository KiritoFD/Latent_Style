#!/bin/bash
# Remote check script - run on remote WSL
echo "=== Running processes ==="
ps aux | grep -E 'python|run\.py' | grep -v grep || echo "none"

echo "=== nohup.out ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/src/nohup.out 2>/dev/null || echo "not_found"

echo "=== train.log ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/src/train.log 2>/dev/null || echo "not_found"

echo "=== Output dirs ==="
ls -d /mnt/i/Github/Latent_Style/SchrodingerBridge/outputs/620_nswd_* 2>/dev/null || echo "no_output_dirs"

echo "=== Latest logs ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/logs/ 2>/dev/null | tail -10 || echo "no_logs"

echo "=== GPU ==="
nvidia-smi 2>/dev/null || echo "no_nvidia"

echo "=== Disk ==="
df -h /mnt/i 2>/dev/null || echo "no_disk"