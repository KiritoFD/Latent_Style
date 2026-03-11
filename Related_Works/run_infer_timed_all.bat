@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ==========================================================
rem Timed Inference Runner (Serial, No Parallel)
rem Steps:
rem 1) SDXL-Turbo img2img (timed) -> Related_Works\runs\sdturbo_5x5
rem 2) CUT test.py (timed)        -> Related_Works\runs\cut_5x5\infer_5x5
rem 3) SDEdit multi-strength      -> Related_Works\runs\sdedit_multi
rem ==========================================================

rem Set CLEAN=1 to delete existing outputs before running.
rem Set CLEAN=0 to resume/skip existing files.
set "CLEAN=1"

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul

set "PY_SDT=%REPO_ROOT%\Related_Works\envs\sdturbo\Scripts\python.exe"
set "PY_CUT=%REPO_ROOT%\Related_Works\envs\cut\Scripts\python.exe"
set "PY_SDE=%REPO_ROOT%\Related_Works\envs\sdedit\Scripts\python.exe"

set "SCRIPT_SDT=%REPO_ROOT%\Related_Works\sdturbo_generate_5x5.py"
set "SCRIPT_CUT=%REPO_ROOT%\Related_Works\cut_infer_5x5.py"
set "SCRIPT_SDE=%REPO_ROOT%\Related_Works\sdedit_generate_5x5.py"

set "TEST_DIR=G:\GitHub\Latent_Style\style_data\overfit50"

set "OUT_SDT=%REPO_ROOT%\Related_Works\runs\sdturbo_5x5"
set "OUT_CUT=%REPO_ROOT%\Related_Works\runs\cut_5x5\infer_5x5"
set "RAW_CUT=%REPO_ROOT%\Related_Works\runs\cut_5x5\raw_results"
set "OUT_SDE=%REPO_ROOT%\Related_Works\runs\sdedit_multi"

set "LOG_ROOT=%REPO_ROOT%\Related_Works\runs\benchmark_logs"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"
set "LOG=%LOG_ROOT%\run_latest.log"

echo ========================================================== > "%LOG%"
echo START %DATE% %TIME% >> "%LOG%"
echo CLEAN=%CLEAN% >> "%LOG%"
echo ========================================================== >> "%LOG%"

rem ---- Preconditions ----
if not exist "%PY_SDT%" ( echo [ERR] Missing %PY_SDT% & exit /b 1 )
if not exist "%PY_CUT%" ( echo [ERR] Missing %PY_CUT% & exit /b 1 )
if not exist "%PY_SDE%" ( echo [ERR] Missing %PY_SDE% & exit /b 1 )
if not exist "%SCRIPT_SDT%" ( echo [ERR] Missing %SCRIPT_SDT% & exit /b 1 )
if not exist "%SCRIPT_CUT%" ( echo [ERR] Missing %SCRIPT_CUT% & exit /b 1 )
if not exist "%SCRIPT_SDE%" ( echo [ERR] Missing %SCRIPT_SDE% & exit /b 1 )
if not exist "%TEST_DIR%" ( echo [ERR] Missing %TEST_DIR% & exit /b 1 )

rem ==========================================================
rem Step 1) SD-Turbo
rem ==========================================================
echo. >> "%LOG%"
echo [STEP1] SD-Turbo %DATE% %TIME% >> "%LOG%"
if "%CLEAN%"=="1" (
  if exist "%OUT_SDT%" rmdir /s /q "%OUT_SDT%"
)
mkdir "%OUT_SDT%" 2>nul

echo CMD: "%PY_SDT%" "%SCRIPT_SDT%" --test_dir "%TEST_DIR%" --out_dir "%OUT_SDT%" --size 256 --offload model --attention_slicing max --vae_slicing --num_steps 1 --strength 1.0 --guidance 0.0 --seed 42 >> "%LOG%"
"%PY_SDT%" "%SCRIPT_SDT%" ^
  --test_dir "%TEST_DIR%" ^
  --out_dir "%OUT_SDT%" ^
  --size 256 ^
  --offload model ^
  --attention_slicing max ^
  --vae_slicing ^
  --num_steps 1 ^
  --strength 1.0 ^
  --guidance 0.0 ^
  --seed 42 ^
  >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERR] SD-Turbo failed. See %LOG% >> "%LOG%"
  exit /b 1
)

rem ==========================================================
rem Step 2) CUT timed re-run
rem ==========================================================
echo. >> "%LOG%"
echo [STEP2] CUT %DATE% %TIME% >> "%LOG%"
if "%CLEAN%"=="1" (
  if exist "%OUT_CUT%" rmdir /s /q "%OUT_CUT%"
  if exist "%RAW_CUT%" rmdir /s /q "%RAW_CUT%"
)
mkdir "%OUT_CUT%" 2>nul

echo CMD: "%PY_CUT%" "%SCRIPT_CUT%" --size 256 >> "%LOG%"
"%PY_CUT%" "%SCRIPT_CUT%" ^
  --cut_repo "%REPO_ROOT%\Related_Works\external\CUT" ^
  --python_exe "%PY_CUT%" ^
  --datasets_root "%REPO_ROOT%\Related_Works\runs\cut_5x5\datasets" ^
  --checkpoints_dir "%REPO_ROOT%\Related_Works\runs\cut_5x5\checkpoints" ^
  --out_dir "%OUT_CUT%" ^
  --results_dir "%RAW_CUT%" ^
  --size 256 ^
  >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERR] CUT inference failed. See %LOG% >> "%LOG%"
  exit /b 1
)

rem ==========================================================
rem Step 3) SDEdit multi-strength
rem ==========================================================
echo. >> "%LOG%"
echo [STEP3] SDEdit %DATE% %TIME% >> "%LOG%"
if "%CLEAN%"=="1" (
  if exist "%OUT_SDE%" rmdir /s /q "%OUT_SDE%"
)
mkdir "%OUT_SDE%" 2>nul

echo CMD: "%PY_SDE%" "%SCRIPT_SDE%" --test_dir "%TEST_DIR%" --out_dir "%OUT_SDE%" --size 256 --strengths 0.3,0.6,0.8 --steps 50 --guidance 6.0 --seed 42 >> "%LOG%"
"%PY_SDE%" "%SCRIPT_SDE%" ^
  --test_dir "%TEST_DIR%" ^
  --out_dir "%OUT_SDE%" ^
  --size 256 ^
  --strengths 0.3,0.6,0.8 ^
  --steps 50 ^
  --guidance 6.0 ^
  --seed 42 ^
  >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERR] SDEdit failed. See %LOG% >> "%LOG%"
  exit /b 1
)

echo. >> "%LOG%"
echo DONE %DATE% %TIME% >> "%LOG%"
echo Log: %LOG%
popd >nul
exit /b 0

