@echo off
setlocal EnableExtensions

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"
if "%RUN_ROOT%"=="" set "RUN_ROOT=%~dp0..\outputs\adain_750"

python "%~dp0run_adain_750.py" ^
  --mode "%MODE%" ^
  --profile "%PROFILE%" ^
  --run_root "%RUN_ROOT%"

exit /b %ERRORLEVEL%
