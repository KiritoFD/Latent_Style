@echo off
setlocal
cd /d "%~dp0"
py -3 prepare_weight_sweep_40.py
if errorlevel 1 exit /b %errorlevel%
py -3 run_weight_sweep_40.py %*
exit /b %errorlevel%
