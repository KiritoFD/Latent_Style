@echo off
setlocal EnableExtensions

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"

python "%~dp0run_all_511.py" ^
  --mode "%MODE%" ^
  --profile "%PROFILE%" %*

exit /b %ERRORLEVEL%
