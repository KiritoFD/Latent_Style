#!/usr/bin/env bash
echo "=== Processes ==="
ps -ef | grep -E 'python|ablation_batch' | grep -v grep
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
echo "=== Last 30 log lines ==="
tail -30 /mnt/i/exp_256_photo2art/_ablation_batch_eval.log
echo ""
echo "=== Eval status ==="
bash /mnt/c/Users/Administrator/_check_ablation_status.sh 2>&1 | tail -10
