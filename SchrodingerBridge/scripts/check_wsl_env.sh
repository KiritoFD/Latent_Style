#!/bin/bash
# Check WSL environment for robust background execution
echo "===WSL VERSION==="
uname -a
echo "===SYSTEMD==="
ps -p 1 -o comm= 2>/dev/null
echo "===SETSID AVAIL==="
which setsid
echo "===SCREEN/TMUX==="
which screen 2>/dev/null || echo "no screen"
which tmux 2>/dev/null || echo "no tmux"
echo "===CURRENT WSL SESSIONS==="
who
echo "===I: FREE SPACE==="
df -h /mnt/i | tail -1
