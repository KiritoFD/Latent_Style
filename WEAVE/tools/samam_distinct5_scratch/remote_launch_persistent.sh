#!/usr/bin/env bash
# Launch SaMam training with WSL keep-alive
# Strategy: tmux session + background keep-alive loop that prevents WSL vm shutdown

SESSION_NAME=samam_train_7k
KEEPALIVE_NAME=wsl_keepalive
TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
KEEPALIVE_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.log
KEEPALIVE_PIDFILE=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.pid

mkdir -p /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

# Step 1: Start a keep-alive daemon that runs forever in background
# This prevents WSL vm from shutting down when SSH session ends
cat > /tmp/wsl_keepalive.sh << 'EOF'
#!/usr/bin/env bash
while true; do
    echo "[$(date -Iseconds)] WSL keep-alive ping" >> /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.log
    sleep 300
done
EOF
chmod +x /tmp/wsl_keepalive.sh

# Kill any existing keepalive
if [ -f "$KEEPALIVE_PIDFILE" ]; then
    kill $(cat "$KEEPALIVE_PIDFILE") 2>/dev/null || true
fi

# Start keepalive in background (fully detached with setsid)
setsid bash /tmp/wsl_keepalive.sh &
echo $! > "$KEEPALIVE_PIDFILE"
echo "=== Keepalive started, PID=$(cat $KEEPALIVE_PIDFILE) ==="

# Step 2: Kill existing tmux session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Step 3: Clear old log
> "$LOG" 2>/dev/null || true

# Step 4: Start new detached tmux session running the training script
tmux new-session -d -s "$SESSION_NAME" "bash $TRAIN_SCRIPT 2>&1 | tee $LOG"

echo "=== tmux session created ==="
tmux list-sessions
echo ""

# Step 5: Wait for training to start and verify it's running
echo "=== Waiting 60s for training to initialize ==="
sleep 60

echo "=== tmux pane output (last 40 lines) ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -40
echo ""
echo "=== train process ==="
ps aux | grep -E "train_SaMam|python.*train" | grep -v grep | head -5
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo ""
echo "=== checkpoints saved ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints/ 2>/dev/null | head -10
echo "=== DONE ==="
