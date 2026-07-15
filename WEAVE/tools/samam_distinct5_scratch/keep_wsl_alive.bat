@echo off
REM Keep WSL alive by starting a 90-second sleep process every minute
REM This ensures there's always at least one WSL process running (overlapping)
REM which prevents WSL VM from shutting down
echo [%date% %time%] PING >> C:\Users\Administrator\keepalive_history.log
start "" /b wsl -d Ubuntu-22.04 -e bash -c "sleep 90" 2>nul
exit /b 0
