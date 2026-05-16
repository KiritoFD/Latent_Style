@echo off
setlocal
cd /d "%~dp0"

REM Refresh configs and registry only. This does not start training.
py -3 run_ablation_7epoch.py --prepare_only

endlocal
