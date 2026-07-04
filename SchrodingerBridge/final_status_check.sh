#!/bin/bash
echo "=== Experiment Status Check ==="
echo "Date: $(date)"
echo ""

# Check tmux session
if tmux has-session -t phase2v2 2>/dev/null; then
    echo "✓ Tmux session 'phase2v2' is RUNNING"
else
    echo "✗ Tmux session 'phase2v2' is NOT running"
fi

echo ""
# Show last 15 lines of log
echo "=== Latest Log Output ==="
tail -15 /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/phase2_v2_master.log 2>/dev/null || echo "(Log file not found)"

echo ""
# Check if process is alive
echo "=== Process Check ==="
pgrep -f "run_phase2_v2.sh" > /dev/null && echo "✓ Experiment process is running" || echo "✗ No experiment process found"