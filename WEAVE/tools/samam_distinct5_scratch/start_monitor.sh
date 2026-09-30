#!/usr/bin/env bash
# Start the loop monitor via setsid (survives SSH disconnect)
# Also acts as a WSL keepalive process

MONITOR_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_loop_monitor.sh
PROGRESS_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/progress.log

# Kill any existing monitor
pkill -f "remote_loop_monitor" 2>/dev/null || true

# Start monitor via setsid (fully detached)
setsid bash "$MONITOR_SCRIPT" &
MONITOR_PID=$!
echo "MONITOR_PID=$MONITOR_PID"
sleep 2

# Verify monitor is running
if kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "MONITOR_ALIVE=YES"
else
    echo "MONITOR_ALIVE=NO - checking via pgrep"
    pgrep -fa "remote_loop_monitor"
fi

# Show current training status
echo ""
echo "=== CURRENT STATUS ==="
echo "Timestamp: $(date -Iseconds)"
echo "Train process: $(pgrep -f 'train_SaMam' | head -1)"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "Tmux: $(tmux ls 2>&1)"
echo "Checkpoints: $(ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/ 2>/dev/null | wc -l)"
echo "Last train log line: $(tail -1 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>/dev/null | tr -d '\r')"
