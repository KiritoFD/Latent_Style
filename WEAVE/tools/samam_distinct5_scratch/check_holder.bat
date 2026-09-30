@echo off
echo === WSL.EXE PROCESSES ===
tasklist /fi "imagename eq wsl.exe"
echo.
echo === WSL STATE ===
wsl -l -v
echo.
echo === HOLDER LOG ===
type C:\Users\Administrator\wsl_holder.log 2>nul
echo.
echo === HOLDER TASK STATUS ===
schtasks /Query /TN WSL_Holder /FO LIST | findstr "状态"
exit /b 0
