#!/bin/bash
# Check ablation progress on remote
echo "=== Ablation experiment status ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
total=0
completed=0
pending=0
failed=0

for exp_dir in exp_ablation_620/*/; do
    if [ -d "$exp_dir" ]; then
        total=$((total+1))
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/full_eval/epoch_0003/summary.json" ]; then
            completed=$((completed+1))
        elif [ -f "$exp_dir/epoch_0003.pt" ] || [ -f "$exp_dir/checkpoint/epoch_0003.pt" ]; then
            pending=$((pending+1))
            echo "PENDING: $exp_name"
        else
            failed=$((failed+1))
            echo "NO_CKPT: $exp_name"
        fi
    fi
done

echo ""
echo "Total: $total, Completed: $completed, Pending: $pending, No-ckpt: $failed"

echo ""
echo "=== Running processes ==="
ps aux | grep -E "run_evaluation|ablation_batch" | grep -v grep | head -5

echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
