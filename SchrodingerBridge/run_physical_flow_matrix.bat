@echo off
setlocal
cd /d "%~dp0"

echo Running 16-run Retinex/PDE physical flow matrix...
echo Root: %CD%
echo.
echo Default mode trains 16 configs for 8 epochs and evaluates epochs 4, 6, 8.
echo Existing checkpoints and eval summaries are resumed or skipped automatically.
echo.

if "%LANCET_BATCH_SIZE%"=="" set LANCET_BATCH_SIZE=64
if "%LANCET_EVAL_BATCH_SIZE%"=="" set LANCET_EVAL_BATCH_SIZE=6
echo Train batch: %LANCET_BATCH_SIZE%
echo Eval batch: %LANCET_EVAL_BATCH_SIZE%
echo.

python tools\experiments\run_physical_flow_matrix.py %*

echo.
echo Physical flow matrix finished.
echo Summary: exp\physical_flow_matrix\physical_frontier.csv
endlocal
