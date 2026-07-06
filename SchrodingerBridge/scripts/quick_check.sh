#!/bin/bash
echo "PID_CHECK:"
ps -ef | grep "run.py" | grep -v grep | head -3
echo "GPU_CHECK:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "LOG_TAIL:"
tail -3 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log | tr '\r' '\n' | tail -3
echo "LOG_MTIME:"
stat -c '%Y %y' /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
echo "NOW:"
date '+%s %Y-%m-%d %H:%M:%S'
