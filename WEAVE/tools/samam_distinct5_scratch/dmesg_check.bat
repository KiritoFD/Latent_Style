@echo off
echo === DMESG FULL (last 40) ===
wsl -d Ubuntu-22.04 -e bash -c "dmesg 2>/dev/null | tail -40"
echo.
echo === WSL STATE ===
wsl -l -v
echo.
echo === KEEPALIVE HISTORY ===
type C:\Users\Administrator\keepalive_history.log 2>nul
echo.
echo === TASKLIST WSL ===
tasklist /fi "imagename eq wsl.exe" /fo csv 2>nul
echo.
echo === WSL PROCESSES INSIDE ===
wsl -d Ubuntu-22.04 -e bash -c "ps aux 2>/dev/null | head -30"
exit /b 0
