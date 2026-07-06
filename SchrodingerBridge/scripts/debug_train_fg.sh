#!/bin/bash
# Run training in foreground WITHOUT tee/pipe to capture any crash error directly.
# Kill after 90 seconds if still running.
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CONFIG=$REPO/configs/630_latent_256_photo2art.json
LOG=/mnt/i/exp_256_photo2art/_train_debug.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[DEBUG] Running training in foreground, will kill after 90s"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

# Run directly, no tee, no pipe - capture ALL output including stderr
# Use a background process with a killer
( "$PYTHON" -u "$REPO/run.py" --config "$CONFIG" > "$LOG" 2>&1 ) &
PID=$!
echo "PID=$PID"

# Wait up to 90 seconds
for i in $(seq 1 90); do
    sleep 1
    if ! kill -0 $PID 2>/dev/null; then
        echo "[DEBUG] Process exited after ${i}s"
        break
    fi
done

# If still running, kill it
if kill -0 $PID 2>/dev/null; then
    echo "[DEBUG] Still running after 90s, killing..."
    kill -9 $PID
    echo "[DEBUG] Process was still alive (good sign)"
else
    wait $PID
    RC=$?
    echo "[DEBUG] Process exit code: $RC"
fi

echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "===LAST 50 LINES OF LOG==="
tail -50 "$LOG"
