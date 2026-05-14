@echo off
setlocal EnableExtensions

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"

echo This queue runs missing main-table baselines serially.
echo It assumes preflight/smoke has already passed.

echo [1/4] CAST
set "RUN_ROOT=%~dp0..\outputs\cast_750"
call "%~dp0run_cast_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [2/4] AesFA
set "RUN_ROOT=%~dp0..\outputs\aesfa_750"
call "%~dp0run_aesfa_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [3/4] AesPA-Net
set "RUN_ROOT=%~dp0..\outputs\aespa_750"
call "%~dp0run_aespa_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [4/4] StyTR2
set "RUN_ROOT=%~dp0..\outputs\stytr2_750"
call "%~dp0run_stytr2_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Missing-baseline full queue finished.
exit /b 0
