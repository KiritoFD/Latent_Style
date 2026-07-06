#!/bin/bash
# Launcher: starts abl512 v3 batch training in background with nohup
# This script avoids quote-nesting issues when called via SSH
REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd "$REPO" || exit 1
mkdir -p logs
# Kill any existing abl512 batch
pkill -f "run_abl512_v3.sh" 2>/dev/null || true
sleep 1
# Launch in background
nohup bash scripts/run_abl512_v3.sh > logs/abl512_v3_batch.log 2>&1 &
PID=$!
echo "Launched abl512 v3 batch training"
echo "  PID: $PID"
echo "  Log: $REPO/logs/abl512_v3_batch.log"
sleep 2
# Verify it's running
if kill -0 $PID 2>/dev/null; then
    echo "  Status: RUNNING"
else
    echo "  Status: FAILED to start"
    echo "  Last 20 lines of log:"
    tail -20 logs/abl512_v3_batch.log 2>/dev/null
fi
