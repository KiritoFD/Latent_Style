#!/bin/bash
# 重启 B2 POC 训练 (在远程 WSL 中执行)
set -e
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

echo "=== Config data section ==="
python3 -c "
import json
with open('configs/620_spectral_poc.json') as f:
    cfg = json.load(f)
d = cfg['data']
print('dino_cache_path:', repr(d.get('dino_cache_path')))
print('dino_cache_required:', d.get('dino_cache_required'))
print('pairing_cache_path:', d.get('pairing_cache_path'))
print('latent_cache_dir:', d.get('latent_cache_dir'))
print('num_workers:', cfg['training'].get('num_workers'))
print('persistent_workers:', cfg['training'].get('persistent_workers'))
"

echo ""
echo "=== Kill old tmux session ==="
tmux kill-session -t b2_poc 2>/dev/null || echo "No existing session"

echo ""
echo "=== Start new tmux session ==="
mkdir -p exp/620_spectral_poc
tmux new-session -d -s b2_poc "cd /mnt/i/Github/Latent_Style/SchrodingerBridge && PYTHONUNBUFFERED=1 python3 run.py --config configs/620_spectral_poc.json 2>&1 | tee exp/620_spectral_poc/train.log"
echo "TMUX_STARTED"

echo ""
echo "=== Sessions ==="
tmux list-sessions
