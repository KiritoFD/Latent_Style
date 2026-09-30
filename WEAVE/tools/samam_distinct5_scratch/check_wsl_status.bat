@echo off
echo === WSL LIST ===
wsl -l -v
echo === WSL TEST ===
wsl -d Ubuntu-22.04 -e bash -c "echo WSL_OK; whoami; nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
echo === KEEPALIVE LOG ===
wsl -d Ubuntu-22.04 -e bash -c "cat /tmp/wsl_keepalive.log 2>/dev/null | tail -5"
exit /b 0
