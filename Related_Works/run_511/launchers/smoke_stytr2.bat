@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

python run_511\run_stytr2_750.py ^
  --mode smoke ^
  --profile 4g ^
  --run_root "%CD%\run_511\outputs\stytr2_smoke"

exit /b %ERRORLEVEL%
