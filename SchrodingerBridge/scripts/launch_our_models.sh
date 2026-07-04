#!/usr/bin/env bash
# 启动器：后台运行主脚本
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
mkdir -p /mnt/i/exp_our_models_eval/logs
nohup bash scripts/run_our_models_eval.sh > /mnt/i/exp_our_models_eval/main_run.log 2>&1 < /dev/null &
disown
echo "PID=$!"
sleep 3
echo "---FIRST_LOG---"
head -30 /mnt/i/exp_our_models_eval/main_run.log 2>/dev/null
