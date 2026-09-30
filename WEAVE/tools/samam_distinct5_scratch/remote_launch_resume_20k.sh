#!/usr/bin/env bash
set -uo pipefail

# Launcher: resume train 7k->20k, then eval all checkpoints with HF CLIP
# Runs in tmux for persistence. WSL keepalive enabled.

SESSION_NAME=samam_resume_20k
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch
TRAIN_SCRIPT=$SCRIPT_DIR/remote_resume_train_20k.sh
EVAL_SCRIPT=$SCRIPT_DIR/remote_run_curve_eval_hf.sh
LOG_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

# WSL keepalive (prevents vmIdleTimeout shutdown during long training)
nohup bash -c 'while true; do sleep 3600; done' >/dev/null 2>&1 &
KEEPALIVE_PID=$!
echo "WSL keepalive PID: $KEEPALIVE_PID"

# Kill existing tmux session if any
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create combined run script
COMBINED=/tmp/samam_resume_and_eval.sh
cat > "$COMBINED" << 'INNEREOF'
#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch
TRAIN_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train_resume_20k.log
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log

echo "========================================" | tee "$TRAIN_LOG"
echo "PHASE 1: Resume training 7k -> 20k" | tee -a "$TRAIN_LOG"
echo "START=$(date -Iseconds)" | tee -a "$TRAIN_LOG"
echo "========================================" | tee -a "$TRAIN_LOG"

bash "$SCRIPT_DIR/remote_resume_train_20k.sh" >> "$TRAIN_LOG" 2>&1
TRAIN_RC=$?
echo "TRAIN_EXIT_CODE=$TRAIN_RC" | tee -a "$TRAIN_LOG"
echo "TRAIN_END=$(date -Iseconds)" | tee -a "$TRAIN_LOG"

if [ $TRAIN_RC -ne 0 ]; then
    echo "ERROR: Training failed (rc=$TRAIN_RC), skipping eval" | tee -a "$TRAIN_LOG"
    exit $TRAIN_RC
fi

echo "========================================" | tee "$EVAL_LOG"
echo "PHASE 2: HF-CLIP eval all checkpoints" | tee -a "$EVAL_LOG"
echo "START=$(date -Iseconds)" | tee -a "$EVAL_LOG"
echo "========================================" | tee -a "$EVAL_LOG"

bash "$SCRIPT_DIR/remote_run_curve_eval_hf.sh" >> "$EVAL_LOG" 2>&1
EVAL_RC=$?
echo "EVAL_EXIT_CODE=$EVAL_RC" | tee -a "$EVAL_LOG"
echo "EVAL_END=$(date -Iseconds)" | tee -a "$EVAL_LOG"

echo "ALL_DONE rc=$EVAL_RC"
INNEREOF
chmod +x "$COMBINED"

# Launch in tmux
tmux new-session -d -s "$SESSION_NAME" "bash $COMBINED"
sleep 3

echo "=== tmux session launched ==="
tmux ls 2>/dev/null
echo ""
echo "=== session preview ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -20
echo ""
echo "=== monitor commands ==="
echo "  tmux attach -t $SESSION_NAME"
echo "  tail -f $LOG_DIR/train_resume_20k.log"
echo "  tail -f $LOG_DIR/eval_hf.log"
