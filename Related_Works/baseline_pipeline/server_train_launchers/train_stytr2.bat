@echo off
call "%~dp0common.bat" || exit /b %ERRORLEVEL%

"%PYTHON_BIN%" Related_Works\baseline_pipeline\scripts\train_new_baselines.py ^
  --baselines stytr2 ^
  --run_root "%RUN_ROOT%" ^
  --python "%PYTHON_BIN%" ^
  --images_per_style "%IMAGES_PER_STYLE%" ^
  --batch_size "%BATCH_SIZE%" ^
  --load_size "%LOAD_SIZE%" ^
  --crop_size "%CROP_SIZE%" ^
  --aesfa_iters "%AESFA_ITERS%" ^
  --stytr2_iters "%STYTR2_ITERS%" ^
  > "%RUN_ROOT%\logs\launcher_stytr2.log" 2>&1

set "RC=%ERRORLEVEL%"
type "%RUN_ROOT%\logs\launcher_stytr2.log"
exit /b %RC%
