#!/usr/bin/env bash
set -uo pipefail

SESSION_NAME=samam_eval
EVAL_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_curve_eval.sh
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval.log

# Ensure keepalive running
KEEP_PID=$(pgrep -f "wsl_keepalive" | head -1)
if [ -z "$KEEP_PID" ]; then
    cat > /tmp/wsl_keepalive_eval.sh << 'EOF'
#!/usr/bin/env bash
while true; do
    echo "[$(date -Iseconds)] eval keep-alive ping" >> /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive_eval.log
    sleep 60
done
EOF
    chmod +x /tmp/wsl_keepalive_eval.sh
    setsid bash /tmp/wsl_keepalive_eval.sh &
    echo "Keepalive started for eval"
else
    echo "Keepalive already running (PID=$KEEP_PID)"
fi

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

if [ -f "$LOG" ] && [ -s "$LOG" ]; then
    mv "$LOG" "$LOG.bak.$(date +%s)" 2>/dev/null || true
fi

tmux new-session -d -s "$SESSION_NAME" "bash $EVAL_SCRIPT 2>&1 | tee $LOG"

echo "=== Eval tmux session created ==="
tmux list-sessions
echo ""
echo "=== Waiting 30s for eval to initialize ==="
sleep 30
echo "=== tmux pane output (last 20 lines) ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -20
echo ""
echo "=== eval process ==="
ps aux | grep -E "eval_samam|python.*eval" | grep -v grep | head -3
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo "=== LAUNCH_TIME=$(date -Iseconds) ==="
