@echo off
schtasks /Create /TN WSL_KeepAlive /TR "C:\Users\Administrator\keep_wsl_alive.bat" /SC MINUTE /MO 1 /RU SYSTEM /F
schtasks /Run /TN WSL_KeepAlive
echo TASK_STATUS:
schtasks /Query /TN WSL_KeepAlive /FO LIST
exit /b 0
