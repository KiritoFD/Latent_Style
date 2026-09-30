#!/bin/bash
# Clean old checkpoints and restart
set -euo pipefail

EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke"

echo "=== Cleaning old checkpoints ==="
rm -f "$EXP_DIR"/epoch_*.pt 2>/dev/null || true
rm -f "$EXP_DIR"/optimizer_*.pt 2>/dev/null || true
rm -f "$EXP_DIR"/scheduler_*.pt 2>/dev/null || true
rm -f "$EXP_DIR"/*.pt 2>/dev/null || true
echo "Cleaned"

echo "=== Restarting training ==="
SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/src"
cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR"

# Clear old logs
rm -f "$EXP_DIR/train.log" 2>/dev/null || true

nohup python3 run.py \
    --config "$EXP_DIR/config.json" \
    > "$EXP_DIR/train.log" 2>&1 &
PID=$!
echo "Launched PID: $PID"

sleep 15
if kill -0 $PID 2>/dev/null; then
    echo "PROCESS RUNNING (PID=$PID)"
    echo "=== First 40 lines of log ==="
    head -40 "$EXP_DIR/train.log" 2>/dev/null || echo "(log empty)"
else
    echo "PROCESS DIED!"
    cat "$EXP_DIR/train.log" 2>/dev/null || echo "(no log)"
fi