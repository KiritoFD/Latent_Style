@echo off
REM Delete SYSTEM-based holder task
schtasks /Delete /TN WSL_Holder /F 2>nul
REM Create holder task as current user (administrator) - SYSTEM can't access WSL distros
schtasks /Create /TN WSL_Holder /TR "C:\Users\Administrator\wsl_persistent_holder.bat" /SC ONCE /ST 00:00 /F
REM Run it now
schtasks /Run /TN WSL_Holder
echo === HOLDER TASK STATUS ===
schtasks /Query /TN WSL_Holder /FO LIST
exit /b 0
