#!/bin/bash
echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi failed"
echo ""
echo "=== Running python processes ==="
ps aux | grep python | grep -v grep | head -10
