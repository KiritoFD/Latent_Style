#!/usr/bin/env bash
set -uo pipefail
pkill -f "run.py.*630_pixel_256_photo2art" 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo "===GPU CLEANED==="
