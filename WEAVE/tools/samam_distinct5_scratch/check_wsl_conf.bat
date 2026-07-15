@echo off
echo === WSL.CONF ===
wsl -d Ubuntu-22.04 -e bash -c "cat /etc/wsl.conf 2>/dev/null; echo '---'; ls -la /etc/wsl.conf 2>/dev/null"
echo.
echo === SYSTEMD STATUS ===
wsl -d Ubuntu-22.04 -e bash -c "ps -p 1 -o comm= 2>/dev/null; systemctl is-system-running 2>/dev/null; systemctl --user status 2>&1 | head -10"
echo.
echo === USER INFO ===
wsl -d Ubuntu-22.04 -e bash -c "id; whoami; echo HOME=$HOME"
echo.
echo === CHECK IF SLEEP SURVIVES ===
wsl -d Ubuntu-22.04 -e bash -c "pgrep -fa sleep 2>/dev/null; echo SLEEP_CHECK_DONE"
exit /b 0
