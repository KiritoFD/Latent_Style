@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"

echo ============================================================
echo   run_511: Serial train + 750-inference for all baselines
echo   Profile: %PROFILE%   Mode: %MODE%
echo   Baselines: StyTR-2, AdaIN, AesFA, AesPA-Net
echo ============================================================

python run_511\run_all_511.py ^
  --mode "%MODE%" ^
  --profile "%PROFILE%" ^
  --baselines stytr2 adain aesfa aespa

exit /b %ERRORLEVEL%
