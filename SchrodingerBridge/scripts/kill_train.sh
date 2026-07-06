#!/bin/bash
# Kill the current training/eval process and verify it's dead
echo "[INFO] Killing training/eval processes..."
pkill -f "run.py.*630_latent_256_photo2art" 2>/dev/null || true
pkill -f "run_evaluation.py" 2>/dev/null || true
sleep 3

echo "===REMAINING PROCESSES==="
ps -ef | grep -E "run.py|run_evaluation" | grep -v grep || echo "ALL DEAD"
echo ""
echo "===GPU==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
