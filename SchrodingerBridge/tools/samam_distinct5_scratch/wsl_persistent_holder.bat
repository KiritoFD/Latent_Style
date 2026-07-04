@echo off
REM Persistent WSL holder - loops forever, restarting WSL if it dies
REM This creates a long-running wsl.exe process that prevents WSL VM shutdown
:loop
echo [%date% %time%] Starting WSL holder >> C:\Users\Administrator\wsl_holder.log
wsl -d Ubuntu-22.04 -e bash -c "while true; do sleep 3600; done"
echo [%date% %time%] WSL holder exited, restarting in 5s >> C:\Users\Administrator\wsl_holder.log
ping -n 6 127.0.0.1 >nul
goto loop
