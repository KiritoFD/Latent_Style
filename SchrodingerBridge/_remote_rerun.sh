#!/bin/bash
# 复制 src/run.py 到项目目录 + 重启训练
set -e
cp /mnt/c/Users/administrator/run.py /mnt/i/Github/Latent_Style/SchrodingerBridge/src/run.py
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

echo "=== Verify run.py fix ==="
grep -n "620_spectral_ode" src/run.py | head -5

echo ""
echo "=== Syntax check ==="
python3 -c "import ast; ast.parse(open('src/run.py').read()); print('SYNTAX_OK')"

echo ""
echo "=== Restart B2 POC training ==="
mkdir -p exp/620_spectral_poc
tmux kill-session -t b2_poc 2>/dev/null || true
tmux new-session -d -s b2_poc "cd /mnt/i/Github/Latent_Style/SchrodingerBridge && PYTHONUNBUFFERED=1 python3 run.py --config configs/620_spectral_poc.json 2>&1 | tee exp/620_spectral_poc/train.log"
echo "TMUX_STARTED"
echo "=== DONE ==="
