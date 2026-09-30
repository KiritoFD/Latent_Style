#!/usr/bin/env bash
echo "=== screen ==="
which screen 2>/dev/null && screen --version || echo "no screen"
echo "=== tmux ==="
which tmux 2>/dev/null && tmux -V || echo "no tmux"
echo "=== nohup ==="
which nohup
echo "=== setsid ==="
which setsid 2>/dev/null || echo "no setsid"
echo "=== DONE ==="
