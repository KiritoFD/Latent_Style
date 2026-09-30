@echo off
:: Launch StyTR-2 inference on all 3 datasets via schtasks
:: This bat wraps the PowerShell runner with absolute python path and log redirection
setlocal
set PYTHON=C:\Program Files\Python312\python.exe
set STYTR2_ROOT=I:\StyTR2
set LOG=I:\StyTR2\stytr2_run.log

echo === StyTR-2 inference start: %DATE% %TIME% === > "%LOG%"
cd /d "%STYTR2_ROOT%"

powershell -ExecutionPolicy Bypass -File "%STYTR2_ROOT%\run_stytr2_all.ps1" >> "%LOG%" 2>&1

echo === StyTR-2 inference end: %DATE% %TIME% === >> "%LOG%"
endlocal
