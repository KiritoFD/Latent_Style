@echo off
setlocal
cd /d "%~dp0"
py -3 prepare_theory_switch_validation.py
if errorlevel 1 exit /b %errorlevel%
py -3 run_theory_switch_validation.py %*
exit /b %errorlevel%
