@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ==========================================================
rem SDEdit rerun + full_eval (KID enabled)
rem Strengths: 0.10, 0.20, 0.35, 0.40
rem Goal: use more VRAM for faster inference (batching, no low-vram offload)
rem ==========================================================

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%" >nul

set "PY_SDE=%REPO_ROOT%\Related_Works\envs\sdedit\Scripts\python.exe"
set "PY_EVAL=%REPO_ROOT%\Cycle-NCE\.venv\Scripts\python.exe"

set "SCRIPT_SDE=%REPO_ROOT%\Related_Works\sdedit_generate_5x5.py"
set "SCRIPT_FE=%REPO_ROOT%\Related_Works\run_full_eval_external.py"
set "EVAL_SCRIPT=%REPO_ROOT%\Cycle-NCE\src\utils\run_evaluation.py"

set "TEST_DIR=%REPO_ROOT%\style_data\overfit50"
set "OUT_SDE=%REPO_ROOT%\Related_Works\runs\sdedit_multi"
set "STRENGTHS=0.1,0.2,0.35,0.4"

rem ---- User knobs ----
set "SDE_BATCH=4"
set "SDE_STEPS=50"
set "SDE_GUIDANCE=6.0"
set "SEED=42"

set "EVAL_FORCE_REGEN=1"
set "EVAL_ENABLE_KID=1"
set "EVAL_BATCH_SIZE=8"
set "EVAL_LPIPS_CHUNK=1"
set "EVAL_MAX_REF_COMPARE=50"
set "EVAL_MAX_REF_CACHE=256"

if not exist "%PY_SDE%" ( echo [ERR] Missing %PY_SDE% & exit /b 1 )
if not exist "%PY_EVAL%" ( echo [ERR] Missing %PY_EVAL% & exit /b 1 )
if not exist "%SCRIPT_SDE%" ( echo [ERR] Missing %SCRIPT_SDE% & exit /b 1 )
if not exist "%SCRIPT_FE%" ( echo [ERR] Missing %SCRIPT_FE% & exit /b 1 )
if not exist "%EVAL_SCRIPT%" ( echo [ERR] Missing %EVAL_SCRIPT% & exit /b 1 )
if not exist "%TEST_DIR%" ( echo [ERR] Missing %TEST_DIR% & exit /b 1 )

echo [1/2] Inference (SDEdit) -> %OUT_SDE%
if exist "%OUT_SDE%" rmdir /s /q "%OUT_SDE%"
mkdir "%OUT_SDE%" 2>nul

"%PY_SDE%" "%SCRIPT_SDE%" ^
  --test_dir "%TEST_DIR%" ^
  --out_dir "%OUT_SDE%" ^
  --size 256 ^
  --strengths %STRENGTHS% ^
  --steps %SDE_STEPS% ^
  --guidance %SDE_GUIDANCE% ^
  --seed %SEED% ^
  --batch %SDE_BATCH%
if errorlevel 1 ( echo [ERR] SDEdit inference failed. & exit /b 1 )

echo [2/2] Full eval (per strength) with KID
for %%S in (0.10 0.20 0.35 0.40) do (
  set "STR_DIR=%OUT_SDE%\str_%%S"
  if exist "!STR_DIR!\images" (
    echo Evaluating !STR_DIR!
    if "%EVAL_FORCE_REGEN%"=="1" (
      set "FORCE=--force_regen"
    ) else (
      set "FORCE="
    )
    if "%EVAL_ENABLE_KID%"=="1" (
      set "KID=--enable_kid"
    ) else (
      set "KID="
    )
    "%PY_EVAL%" "%SCRIPT_FE%" ^
      --eval_py "%PY_EVAL%" ^
      --eval_script "%EVAL_SCRIPT%" ^
      --test_dir "%TEST_DIR%" ^
      --out_dir "!STR_DIR!" ^
      --batch_size %EVAL_BATCH_SIZE% ^
      --eval_lpips_chunk_size %EVAL_LPIPS_CHUNK% ^
      --max_ref_compare %EVAL_MAX_REF_COMPARE% ^
      --max_ref_cache %EVAL_MAX_REF_CACHE% ^
      !FORCE! ^
      !KID!
    if errorlevel 1 ( echo [ERR] Full eval failed for !STR_DIR! & exit /b 1 )
  ) else (
    echo [WARN] Missing images folder: !STR_DIR!\images
  )
)

echo DONE
popd >nul
exit /b 0
