#!/bin/bash
# Dry run + launch training on remote
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
CONFIG="$EXP_DIR/620_nswd_gate03_smoke/config.json"

echo "=== Step 1: Clear stale .pyc cache ==="
find "$SRC_DIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Done"

echo "=== Step 2: Dry run config load ==="
cd "$SRC_DIR/src"
export PYTHONPATH="$SRC_DIR/src"
python3 -c "
import json, sys
sys.path.insert(0, '$SRC_DIR/src')
from config_schema import ExperimentConfig
with open('$CONFIG') as f:
    cfg = json.load(f)
c = ExperimentConfig.from_mapping(cfg)
print('Config loaded OK')
print('swd_noise_sigma:', c.bridge.swd_noise_sigma)
print('num_epochs:', c.training.num_epochs)
print('latent_channels:', c.model.latent_channels)
" 2>&1 || true
echo "Dry run done"

echo "=== Step 3: Launch training ==="
nohup python3 run.py \
    --config "$CONFIG" \
    > "$EXP_DIR/620_nswd_gate03_smoke/train.log" 2>&1 &
PID=$!
echo "Launched PID: $PID"

sleep 10
if kill -0 $PID 2>/dev/null; then
    echo "PROCESS RUNNING (PID=$PID)"
    echo "=== First 30 lines of log ==="
    head -30 "$EXP_DIR/620_nswd_gate03_smoke/train.log" 2>/dev/null || echo "(log empty)"
else
    echo "PROCESS DIED!"
    echo "=== Log content ==="
    cat "$EXP_DIR/620_nswd_gate03_smoke/train.log" 2>/dev/null || echo "(no log)"
fi