#!/bin/bash
echo "=== GPU check ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "=== Process check ==="
ps aux | grep python | grep -v grep

echo ""
echo "=== Eval log tail ==="
cat /tmp/eval_fc_sb_log.txt 2>/dev/null | tail -20
