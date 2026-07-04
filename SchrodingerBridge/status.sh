#!/bin/bash
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo ""
echo "=== Python processes ==="
ps aux 2>/dev/null | grep python | grep -v grep || echo "No python processes"

echo ""
echo "=== Eval log ==="
cat /tmp/eval_fc_sb_log2.txt 2>/dev/null | tail -30 || echo "No log found"

echo ""
echo "=== Done at $(date) ==="
