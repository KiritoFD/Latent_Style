#!/bin/bash
sleep 60
echo "===LOG (last 30 lines)==="
tail -30 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
echo ""
echo "===NVIDIA==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
echo "===TMUX SESSIONS==="
tmux list-sessions 2>/dev/null
echo "===PROCESS==="
ps -ef | grep -E "run\.py" | grep -v grep
echo "===LOG SIZE==="
ls -la /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
