@echo off
REM Setup WSL keep-alive by modifying .wslconfig and creating scheduled task
REM This ensures WSL VM never auto-shuts down, keeping tmux sessions alive

set WSLCONFIG=C:\Users\Administrator\.wslconfig
set BACKUP=%WSLCONFIG%.bak

REM Backup existing config
if exist "%WSLCONFIG%" copy "%WSLCONFIG%" "%BACKUP%" >nul

REM Write new config with vmIdleTimeout=-1 (never auto-shutdown)
echo [wsl2] > "%WSLCONFIG%"
echo memory=16GB >> "%WSLCONFIG%"
echo localhostForwarding=true >> "%WSLCONFIG%"
echo swap=20GB >> "%WSLCONFIG%"
echo vmIdleTimeout=-1 >> "%WSLCONFIG%"

echo === New .wslconfig ===
type "%WSLCONFIG%"

REM Terminate WSL to apply new config
echo === Terminating WSL to apply config ===
wsl --terminate
timeout /t 3 /nobreak >nul

REM Create scheduled task to keep WSL alive (run sleep infinity in background)
schtasks /Delete /TN "WSL_KeepAlive" /F 2>nul
schtasks /Create /TN "WSL_KeepAlive" /TR "C:\Windows\System32\wsl.exe -d Ubuntu -- exec sleep infinity" /SC ONCE /ST 00:00 /RU SYSTEM /F
schtasks /Run /TN "WSL_KeepAlive"
echo === KeepAlive task started ===
timeout /t 5 /nobreak >nul

REM Verify WSL is running
echo === WSL status ===
wsl -l -v
echo === DONE ===
