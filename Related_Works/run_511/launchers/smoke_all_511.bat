@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

set "PYTHONHOME="
set "PYTHONPATH="

echo ============================================================
echo   run_511: SMOKE TEST all baselines (1 iter, 1 image each)
echo ============================================================

python run_511\run_all_511.py ^
  --mode smoke ^
  --profile 4g ^
  --baselines stytr2 adain aesfa aespa

exit /b %ERRORLEVEL%
