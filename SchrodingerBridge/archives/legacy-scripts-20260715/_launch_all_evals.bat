@echo off
:: Launch the comprehensive baseline eval script via PowerShell
:: Deployed to I:\launch_all_evals.bat on remote
setlocal
echo === LAUNCH ALL EVALS START: %DATE% %TIME% ===
powershell -ExecutionPolicy Bypass -File "I:\run_all_baseline_evals.ps1"
echo === LAUNCH ALL EVALS END: %DATE% %TIME% ===
endlocal
