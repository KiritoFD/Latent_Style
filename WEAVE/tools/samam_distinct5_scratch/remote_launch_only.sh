#!/usr/bin/env bash
# Launch SaMam training - non-blocking, just start and exit
set -uo pipefail

SESSION_NAME=samam_train_7k
TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
OUT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

mkdir -p "$OUT_DIR"

# Step 1: Start a keep-alive daemon that runs forever in background (extra safety)
cat > /tmp/wsl_keepalive.sh << 'EOF'
#!/usr/bin/env bash
while true; do
    echo "[$(date -Iseconds)] WSL keep-alive ping" >> /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.log
    sleep 60
done
EOF
chmod +x /tmp/wsl_keepalive.sh

# Kill any existing keepalive
if [ -f "$OUT_DIR/keepalive.pid" ]; then
    kill "$(cat "$OUT_DIR/keepalive.pid")" 2>/dev/null || true
fi

# Start keepalive in background (fully detached with setsid)
setsid bash /tmp/wsl_keepalive.sh &
echo $! > "$OUT_DIR/keepalive.pid"
echo "KEEPALIVE_PID=$(cat "$OUT_DIR/keepalive.pid")"

# Step 2: Kill existing tmux session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Step 3: Clear old log (backup first)
if [ -f "$LOG" ] && [ -s "$LOG" ]; then
    mv "$LOG" "$LOG.bak.$(date +%s)" 2>/dev/null || true
fi

# Step 4: Start new detached tmux session running the training script
tmux new-session -d -s "$SESSION_NAME" "bash $TRAIN_SCRIPT 2>&1 | tee $LOG"

echo "TMUX_STARTED=1"
tmux list-sessions 2>&1
echo "LAUNCH_TIME=$(date -Iseconds)"
