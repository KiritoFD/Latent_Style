#!/bin/bash
# Check Windows python availability and current datasets dir
echo "=== Windows python ==="
which python.exe 2>&1
python.exe --version 2>&1
echo ""
echo "=== Windows python path ==="
ls /mnt/c/Users/Administrator/AppData/Local/Programs/Python/ 2>&1 | head -5
echo ""
echo "=== Current /mnt/i/datasets/ contents ==="
ls -la /mnt/i/datasets/ 2>&1 | head -20
echo ""
echo "=== Check if any training is running ==="
ps aux | grep -E "run.py|run_abl512" | grep -v grep
echo ""
echo "=== Disk space ==="
df -h /mnt/i
