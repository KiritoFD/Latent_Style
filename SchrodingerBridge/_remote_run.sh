#!/bin/bash
# 远程执行: 复制测试脚本 + Haar 测试 + 模型 smoke test + 启动训练
set -e
cp /mnt/c/Users/administrator/_remote_test.py /mnt/i/Github/Latent_Style/SchrodingerBridge/_remote_test.py
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

echo "=== Test: Haar + Model + Loss smoke test ==="
python3 _remote_test.py

echo ""
echo "=== Launch B2 POC training ==="
mkdir -p exp/620_spectral_poc
tmux kill-session -t b2_poc 2>/dev/null || true
tmux new-session -d -s b2_poc "cd /mnt/i/Github/Latent_Style/SchrodingerBridge && PYTHONUNBUFFERED=1 python3 run.py --config configs/620_spectral_poc.json 2>&1 | tee exp/620_spectral_poc/train.log"
echo "TMUX_STARTED"
echo "=== DONE ==="
