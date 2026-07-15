#!/bin/bash
# Check training status on remote
set -euo pipefail

LOG_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke"

echo "=== Process status ==="
ps aux | grep -E 'python|run\.py' | grep -v grep || echo "No python processes"

echo ""
echo "=== Log file ==="
ls -la "$LOG_DIR/train.log" 2>/dev/null || echo "No log file"

echo ""
echo "=== Last 50 lines of log ==="
tail -50 "$LOG_DIR/train.log" 2>/dev/null || echo "(no log)"

echo ""
echo "=== Checkpoints ==="
ls -la "$LOG_DIR/"*.pt 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"