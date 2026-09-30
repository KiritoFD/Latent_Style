#!/bin/bash
# Restart training with fixed gate=0.3
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

echo "=== Step 1: Kill old processes ==="
pkill -f "run.py" 2>/dev/null || true
sleep 2
echo "Done"

echo "=== Step 2: Clear stale pyc ==="
find "$SRC_DIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Done"

echo "=== Step 3: Verify config ==="
python3 -c "
import json
CONFIG='$EXP_DIR/620_nswd_gate03_smoke/config.json'
with open(CONFIG) as f:
    c = json.load(f)
print('gate:', c['model']['style_cross_attn_gate_init'])
print('swd_noise_sigma:', c['bridge']['swd_noise_sigma'])
print('endpoint_head_mode:', c['model']['endpoint_head_mode'])
print('transport_prediction_mode:', c['model']['transport_prediction_mode'])
"

echo "=== Step 4: Launch training ==="
cd "$SRC_DIR/src"
export PYTHONPATH="$SRC_DIR/src"

# Rename old log
mv "$EXP_DIR/620_nswd_gate03_smoke/train.log" "$EXP_DIR/620_nswd_gate03_smoke/train_old.log" 2>/dev/null || true

nohup python3 run.py \
    --config "$EXP_DIR/620_nswd_gate03_smoke/config.json" \
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
    cat "$EXP_DIR/620_nswd_gate03_smoke/train.log" 2>/dev/null || echo "(no log)"
fi