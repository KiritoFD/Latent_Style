#!/bin/bash
set -euo pipefail

HOST="administrator@100.115.18.62"
SSH_OPTS="-p 2222 -o LogLevel=ERROR -o StrictHostKeyChecking=no"

echo "=== Checking running processes ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"ps aux | grep -E 'python|run\.py' | grep -v grep\"" 2>&1 || echo "No python processes found"

echo ""
echo "=== Checking nohup.out ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/src/nohup.out 2>/dev/null || echo 'not_found'\"" 2>&1 || true

echo ""
echo "=== Checking train.log ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/src/train.log 2>/dev/null || echo 'not_found'\"" 2>&1 || true

echo ""
echo "=== Checking output dirs ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"ls -d /mnt/i/Github/Latent_Style/SchrodingerBridge/outputs/620_nswd_* 2>/dev/null || echo 'no_output_dirs'\"" 2>&1 || true

echo ""
echo "=== Checking latest logs ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/logs/ 2>/dev/null | tail -10 || echo 'no_logs'\"" 2>&1 || true

echo ""
echo "=== Checking GPU ==="
ssh $SSH_OPTS $HOST "wsl bash -c \"nvidia-smi 2>/dev/null || echo 'no_nvidia'\"" 2>&1 || true