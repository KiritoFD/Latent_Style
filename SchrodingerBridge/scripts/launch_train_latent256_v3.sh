#!/usr/bin/env bash
# Launch latent256 training in a detached tmux session.
LOG=/mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
TMUX_NAME=train_latent256

# Kill any existing tmux session with the same name
tmux kill-session -t "$TMUX_NAME" 2>/dev/null
sleep 1

# Create detached tmux session
mkdir -p /mnt/i/exp_256_photo2art
tmux new-session -d -s "$TMUX_NAME" "bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_latent256_photo2art.sh"

sleep 3
echo "===TMUX SESSIONS==="
tmux list-sessions 2>/dev/null
echo "===PROCESS==="
ps -ef | grep -E "run\.py|train_latent256|tmux" | grep -v grep
echo "===LOG LAST 20 LINES==="
tail -20 "$LOG" 2>/dev/null || echo "LOG EMPTY YET"
