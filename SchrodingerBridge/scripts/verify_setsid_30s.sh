#!/bin/bash
sleep 30
echo "===PROCESS CHECK (30s after launch)==="
PID=$(cat /mnt/i/exp_256_photo2art/_train_latent256.pid 2>/dev/null)
echo "Expected PID: $PID"
ps -p $PID -o pid,stat,comm,args 2>/dev/null || echo "PID $PID DEAD"
echo ""
echo "===ALL PYTHON PROCESSES==="
ps -ef | grep -E "python|run\.py" | grep -v grep
echo ""
echo "===LOG LAST 15 LINES==="
tail -15 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
echo ""
echo "===NVIDIA==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
