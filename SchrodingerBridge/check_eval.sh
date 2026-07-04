#!/bin/bash
echo "=== Process check ===" > /tmp/status.txt
ps aux 2>/dev/null | grep python >> /tmp/status.txt || echo "No python" >> /tmp/status.txt

echo "" >> /tmp/status.txt
echo "=== GPU ===" >> /tmp/status.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> /tmp/status.txt

echo "" >> /tmp/status.txt
echo "=== Eval log tail ===" >> /tmp/status.txt
cat /mnt/c/Users/Administrator/fc_sb_sigma04/full_eval/eval_log.txt 2>/dev/null | tail -20 >> /tmp/status.txt || echo "No log" >> /tmp/status.txt

echo "" >> /tmp/status.txt
echo "=== Done ===" >> /tmp/status.txt
cat /tmp/status.txt
