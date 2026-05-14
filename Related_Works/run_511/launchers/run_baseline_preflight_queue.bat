@echo off
setlocal EnableExtensions

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"

echo [1/4] CAST preflight/smoke
set "MODE=smoke"
call "%~dp0run_cast_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [2/4] AesFA smoke
set "MODE=smoke"
call "%~dp0run_aesfa_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [3/4] AesPA-Net preflight
set "MODE=preflight"
call "%~dp0run_aespa_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [4/4] StyTR2 smoke
set "MODE=smoke"
call "%~dp0run_stytr2_750.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Baseline preflight queue finished.
exit /b 0
