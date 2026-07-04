#!/usr/bin/env bash
# Launcher: 启动串行评估脚本
mkdir -p /mnt/i/exp_our_models_eval
nohup bash /mnt/c/Users/Administrator/run_serial_eval_v2.sh > /mnt/i/exp_our_models_eval/main_v2.log 2>&1 < /dev/null &
disown
echo "PID=$!"
sleep 5
echo "=== First 30 lines of log ==="
head -30 /mnt/i/exp_our_models_eval/main_v2.log 2>/dev/null
echo "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
