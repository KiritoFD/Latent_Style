@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..") do set "REPO_ROOT=%%~fI"

if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"
if "%VRAM_PROFILE%"=="" set "VRAM_PROFILE=7g"
if "%RUN_ROOT%"=="" set "RUN_ROOT=%REPO_ROOT%\Related_Works\runs\server_new_baselines\%VRAM_PROFILE%"

if /I "%VRAM_PROFILE%"=="4g" (
  if "%BATCH_SIZE%"=="" set "BATCH_SIZE=1"
  if "%LOAD_SIZE%"=="" set "LOAD_SIZE=128"
  if "%CROP_SIZE%"=="" set "CROP_SIZE=128"
  if "%IMAGES_PER_STYLE%"=="" set "IMAGES_PER_STYLE=16"
  if "%AESFA_ITERS%"=="" set "AESFA_ITERS=200"
  if "%STYTR2_ITERS%"=="" set "STYTR2_ITERS=200"
  if "%AESPA_ITERS%"=="" set "AESPA_ITERS=200"
  if "%NUM_WORKERS%"=="" set "NUM_WORKERS=0"
  goto profile_ok
)

if /I "%VRAM_PROFILE%"=="7g" (
  if "%BATCH_SIZE%"=="" set "BATCH_SIZE=1"
  if "%LOAD_SIZE%"=="" set "LOAD_SIZE=192"
  if "%CROP_SIZE%"=="" set "CROP_SIZE=192"
  if "%IMAGES_PER_STYLE%"=="" set "IMAGES_PER_STYLE=32"
  if "%AESFA_ITERS%"=="" set "AESFA_ITERS=500"
  if "%STYTR2_ITERS%"=="" set "STYTR2_ITERS=500"
  if "%AESPA_ITERS%"=="" set "AESPA_ITERS=500"
  if "%NUM_WORKERS%"=="" set "NUM_WORKERS=2"
  goto profile_ok
)

if /I "%VRAM_PROFILE%"=="11g" (
  if "%BATCH_SIZE%"=="" set "BATCH_SIZE=2"
  if "%LOAD_SIZE%"=="" set "LOAD_SIZE=256"
  if "%CROP_SIZE%"=="" set "CROP_SIZE=256"
  if "%IMAGES_PER_STYLE%"=="" set "IMAGES_PER_STYLE=64"
  if "%AESFA_ITERS%"=="" set "AESFA_ITERS=1000"
  if "%STYTR2_ITERS%"=="" set "STYTR2_ITERS=1000"
  if "%AESPA_ITERS%"=="" set "AESPA_ITERS=1000"
  if "%NUM_WORKERS%"=="" set "NUM_WORKERS=4"
  goto profile_ok
)

echo Unknown VRAM_PROFILE="%VRAM_PROFILE%". Use 4g, 7g, or 11g.
exit /b 2

:profile_ok
if not exist "%RUN_ROOT%\logs" mkdir "%RUN_ROOT%\logs"

echo [profile] VRAM_PROFILE=%VRAM_PROFILE%
echo [profile] RUN_ROOT=%RUN_ROOT%
echo [profile] BATCH_SIZE=%BATCH_SIZE% LOAD_SIZE=%LOAD_SIZE% CROP_SIZE=%CROP_SIZE%
echo [profile] IMAGES_PER_STYLE=%IMAGES_PER_STYLE% NUM_WORKERS=%NUM_WORKERS%
echo [profile] AESFA_ITERS=%AESFA_ITERS% STYTR2_ITERS=%STYTR2_ITERS% AESPA_ITERS=%AESPA_ITERS%

cd /d "%REPO_ROOT%"
