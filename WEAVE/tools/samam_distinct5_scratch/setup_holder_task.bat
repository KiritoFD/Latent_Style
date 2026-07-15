@echo off
REM Delete old per-minute keepalive task
schtasks /Delete /TN WSL_KeepAlive /F 2>nul
REM Create new ONCE task that runs the persistent holder
REM This task will keep running because the holder loops forever
schtasks /Create /TN WSL_Holder /TR "C:\Users\Administrator\wsl_persistent_holder.bat" /SC ONCE /ST 00:00 /RU SYSTEM /F
REM Run it now
schtasks /Run /TN WSL_Holder
echo === HOLDER TASK STATUS ===
schtasks /Query /TN WSL_Holder /FO LIST
exit /b 0
