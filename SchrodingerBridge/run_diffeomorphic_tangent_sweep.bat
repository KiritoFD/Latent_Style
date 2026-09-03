@echo off
setlocal
cd /d "%~dp0"

echo Running 12-run diffeomorphic tangent sweep...
echo Root: %CD%
echo.
echo Train batch override: %LANCET_BATCH_SIZE%
echo Eval batch override: %LANCET_EVAL_BATCH_SIZE%
echo If unset, the script uses GPU-tier defaults: 8GB=64/8, 12GB=160/16.
echo.

python tools\experiments\run_diffeomorphic_tangent_sweep.py %*

echo.
echo Tangent sweep finished.
echo Summary: exp\diffeomorphic_tangent_sweep\tangent_grid_frontier.csv
endlocal
