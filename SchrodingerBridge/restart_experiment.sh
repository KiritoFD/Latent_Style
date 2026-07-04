#!/bin/bash
echo "=== FC-SB Phase 2 v2 Experiment Restart ==="
echo "Date: $(date)"
echo ""

# Kill existing tmux session if exists
echo "Checking for existing phase2v2 session..."
if tmux has-session -t phase2v2 2>/dev/null; then
    echo "Killing existing phase2v2 session..."
    tmux kill-session -t phase2v2
    echo "✓ Old session killed"
else
    echo "No existing session found"
fi

echo ""
# Check if experiment script exists
EXP_SCRIPT="/home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/run_phase2_v2.sh"
if [ -f "$EXP_SCRIPT" ]; then
    echo "✓ Found experiment script: $EXP_SCRIPT"
else
    echo "✗ Experiment script not found at $EXP_SCRIPT"
    echo "Looking for available scripts..."
    ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/*.sh 2>&1
    exit 1
fi

echo ""
# Start new tmux session with experiment
echo "Starting new phase2v2 session..."
tmux new-session -d -s phase2v2 "cd /home/xy/Latent_Style/SchrodingerBridge && bash exp/p3_remote_10h/run_phase2_v2.sh 2>&1 | tee exp/p3_remote_10h/phase2_v2_master.log"

sleep 2

# Verify session is running
if tmux has-session -t phase2v2 2>/dev/null; then
    echo "✓ Session 'phase2v2' is running"
    echo ""
    echo "=== Initial Log Output ==="
    head -20 /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/phase2_v2_master.log 2>/dev/null || echo "(Log file not yet created)"
else
    echo "✗ Failed to start session"
    exit 1
fi

echo ""
echo "=== Experiment Restart Complete ==="