@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=G:\GitHub\Latent_Style\Related_Works\Dreambooth-Stable-Diffusion-main"
set "PYTHON=python"

set "RUNS_ROOT=G:\GitHub\Latent_Style\Related_Works\runs\dreambooth"
set "JOB_NAME=db_style"

set "BASE_SD_CKPT=G:\GitHub\Latent_Style\models\sd-v1-4-full-ema.ckpt"
set "TRAIN_DATA_ROOT=G:\GitHub\Latent_Style\style_data\train\cezanne"
set "REG_DATA_ROOT=G:\GitHub\Latent_Style\style_data\train\photo"
set "CLASS_WORD=painting"
set "GPU_IDS=0,"

if "%~1"=="" (
  set "CHUNK_STEPS=200"
) else (
  set "CHUNK_STEPS=%~1"
)
if "%~2"=="" (
  set "ROUNDS=1"
) else (
  set "ROUNDS=%~2"
)

if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"

if not exist "%BASE_SD_CKPT%" (
  echo [ERROR] BASE_SD_CKPT not found: %BASE_SD_CKPT%
  exit /b 1
)
if not exist "%TRAIN_DATA_ROOT%" (
  echo [ERROR] TRAIN_DATA_ROOT not found: %TRAIN_DATA_ROOT%
  exit /b 1
)
if not exist "%REG_DATA_ROOT%" (
  echo [ERROR] REG_DATA_ROOT not found: %REG_DATA_ROOT%
  exit /b 1
)

echo [INFO] chunk_steps=%CHUNK_STEPS%
echo [INFO] rounds=%ROUNDS%
echo [INFO] job=%JOB_NAME%
echo [INFO] train_data=%TRAIN_DATA_ROOT%
echo [INFO] reg_data=%REG_DATA_ROOT%

for /L %%R in (1,1,%ROUNDS%) do (
  echo.
  echo [INFO] ===== Round %%R / %ROUNDS% =====

  set "LATEST_LOGDIR="
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$r='%RUNS_ROOT%'; if(Test-Path $r){$d=Get-ChildItem $r -Directory | Where-Object { $_.Name -like '*_%JOB_NAME%' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($d){$d.FullName}}"`) do (
    set "LATEST_LOGDIR=%%I"
  )

  set "LAST_CKPT="
  if defined LATEST_LOGDIR (
    if exist "!LATEST_LOGDIR!\checkpoints\last.ckpt" (
      set "LAST_CKPT=!LATEST_LOGDIR!\checkpoints\last.ckpt"
    )
  )

  if defined LAST_CKPT (
    for /f "usebackq delims=" %%S in (`%PYTHON% -c "import torch; ck=torch.load(r'!LAST_CKPT!', map_location='cpu', weights_only=False); print(int(ck.get('global_step',0)))"`) do (
      set "GLOBAL_STEP=%%S"
    )
    if not defined GLOBAL_STEP set "GLOBAL_STEP=0"
    set /a TARGET_STEPS=!GLOBAL_STEP!+%CHUNK_STEPS%

    echo [INFO] resume logdir: !LATEST_LOGDIR!
    echo [INFO] resume ckpt: !LAST_CKPT!
    echo [INFO] global_step=!GLOBAL_STEP! => target_max_steps=!TARGET_STEPS!

    %PYTHON% "%ROOT%\main.py" ^
      --base "%ROOT%\configs\stable-diffusion\v1-finetune_unfrozen.yaml" ^
      -t ^
      --resume "!LATEST_LOGDIR!" ^
      --actual_resume "!LAST_CKPT!" ^
      --data_root "%TRAIN_DATA_ROOT%" ^
      --reg_data_root "%REG_DATA_ROOT%" ^
      --class_word "%CLASS_WORD%" ^
      --gpus %GPU_IDS% ^
      --max_steps !TARGET_STEPS! ^
      --logdir "%RUNS_ROOT%" ^
      --datadir_in_name False ^
      --no-test True
  ) else (
    set /a TARGET_STEPS=%CHUNK_STEPS%
    echo [INFO] start new run, target_max_steps=!TARGET_STEPS!
    %PYTHON% "%ROOT%\main.py" ^
      --base "%ROOT%\configs\stable-diffusion\v1-finetune_unfrozen.yaml" ^
      -t ^
      -n "%JOB_NAME%" ^
      --actual_resume "%BASE_SD_CKPT%" ^
      --data_root "%TRAIN_DATA_ROOT%" ^
      --reg_data_root "%REG_DATA_ROOT%" ^
      --class_word "%CLASS_WORD%" ^
      --gpus %GPU_IDS% ^
      --max_steps !TARGET_STEPS! ^
      --logdir "%RUNS_ROOT%" ^
      --datadir_in_name False ^
      --no-test True
  )

  if errorlevel 1 (
    echo [ERROR] training failed in round %%R
    exit /b 1
  )
)

echo.
echo [INFO] all rounds completed
exit /b 0

