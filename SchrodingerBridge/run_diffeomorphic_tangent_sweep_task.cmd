@echo off
setlocal
cd /d "%~dp0"

if not exist logs mkdir logs

python tools\experiments\run_diffeomorphic_tangent_sweep.py --force-train %* 1> logs\diffeomorphic_tangent_sweep.out.log 2> logs\diffeomorphic_tangent_sweep.err.log

endlocal
