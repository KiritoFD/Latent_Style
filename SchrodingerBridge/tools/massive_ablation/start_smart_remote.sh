#!/usr/bin/env bash
# Wrapper to start the smart ablation inside a detached tmux session on remote WSL.

SESSION_NAME="smart_ablation"

# Kill existing session if any
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Start new detached session running the ablation script
tmux new-session -d -s "$SESSION_NAME" -c "/mnt/i/Github/Latent_Style/SchrodingerBridge" "bash tools/massive_ablation/run_smart_ablation.sh"

echo "Started tmux session '$SESSION_NAME' running run_smart_ablation.sh"
echo "To attach, run: tmux attach -t $SESSION_NAME"
