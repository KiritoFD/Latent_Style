@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"
if "%RUN_ROOT%"=="" set "RUN_ROOT=%CD%\run_511\outputs\aespa_750"

python run_511\run_aespa_750.py ^
  --mode "%MODE%" ^
  --profile "%PROFILE%" ^
  --run_root "%RUN_ROOT%"

exit /b %ERRORLEVEL%
