@echo off
setlocal
cd /d "%~dp0"

REM Defaults keep the exact training batch/eval batch from the corrected base config.
REM To run more conservatively on a small GPU:
REM   set BATCH_SIZE=32
REM   set EVAL_BATCH_SIZE=6
REM   run_destructive_ablation_7epoch.bat
if "%BATCH_SIZE%"=="" set BATCH_SIZE=0
if "%EVAL_BATCH_SIZE%"=="" set EVAL_BATCH_SIZE=0
if "%ABLATION_ONLY%"=="" (
  py -3 run_ablation_7epoch.py --batch_size %BATCH_SIZE% --eval_batch_size %EVAL_BATCH_SIZE%
) else (
  py -3 run_ablation_7epoch.py --batch_size %BATCH_SIZE% --eval_batch_size %EVAL_BATCH_SIZE% --only %ABLATION_ONLY%
)

endlocal
