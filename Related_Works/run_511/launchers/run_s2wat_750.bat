@echo off
setlocal EnableExtensions

set "PYTHONHOME="
set "PYTHONPATH="

if "%PROFILE%"=="" set "PROFILE=7g"
if "%MODE%"=="" set "MODE=all"
if "%RUN_ROOT%"=="" set "RUN_ROOT=%~dp0..\outputs\s2wat_750_strict"
if "%CHECKPOINT_ROOT%"=="" set "CHECKPOINT_ROOT=%~dp0..\..\baseline_pipeline\checkpoints\s2wat"
if "%EVAL_TASKS%"=="" set "EVAL_TASKS=base guard"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=py -3"

%PYTHON_EXE% "%~dp0run_s2wat_750.py" ^
  --mode "%MODE%" ^
  --profile "%PROFILE%" ^
  --run_root "%RUN_ROOT%" ^
  --checkpoint_root "%CHECKPOINT_ROOT%" ^
  --eval_tasks %EVAL_TASKS% ^
  %EXTRA_ARGS%

exit /b %ERRORLEVEL%
