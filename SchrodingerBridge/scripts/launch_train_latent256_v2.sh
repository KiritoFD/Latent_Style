#!/usr/bin/env bash
# Launch latent256 training in a detached screen session.
# This survives SSH disconnect because screen creates a new session
# independent of the controlling terminal.
LOG=/mnt/i/exp_256_photo2art/_train_latent256_photo2art.log
SCREEN_NAME=train_latent256

# Kill any existing screen with the same name
screen -S "$SCREEN_NAME" -X quit 2>/dev/null
sleep 1

# Create detached screen session running the training script
# -dmS: start detached
# -L: log output to screenlog.* (backup)
# The train script already tees to $LOG
mkdir -p /mnt/i/exp_256_photo2art
screen -dmS "$SCREEN_NAME" -L -Logfile /mnt/i/exp_256_photo2art/_screen_latent256.log bash /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_latent256_photo2art.sh

sleep 3
echo "===SCREEN SESSIONS==="
screen -ls
echo "===PROCESS==="
ps -ef | grep -E "run\.py|train_latent256" | grep -v grep
echo "===LOG FIRST 10 LINES==="
head -10 "$LOG" 2>/dev/null || echo "LOG EMPTY YET"
