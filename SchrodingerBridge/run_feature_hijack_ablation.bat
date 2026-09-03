@echo off
setlocal
cd /d "%~dp0"

echo Running 12-run feature-hijack ablation sweep...
echo Root: %CD%
echo.
echo Default mode trains 12 configs for 8 epochs and evaluates epochs 4, 6, 8.
echo Existing checkpoints and eval summaries are resumed or skipped automatically.
echo.

if "%LANCET_BATCH_SIZE%"=="" set LANCET_BATCH_SIZE=64
if "%LANCET_EVAL_BATCH_SIZE%"=="" set LANCET_EVAL_BATCH_SIZE=6
echo Train batch: %LANCET_BATCH_SIZE%
echo Eval batch: %LANCET_EVAL_BATCH_SIZE%
echo.

python tools\experiments\run_feature_hijack_ablation.py %*

echo.
echo Feature-hijack sweep finished.
echo Summary: exp\feature_hijack_ablation\mechanism_frontier.csv
endlocal
