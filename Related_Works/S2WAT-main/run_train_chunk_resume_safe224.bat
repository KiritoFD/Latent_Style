@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=G:\GitHub\Latent_Style\Related_Works\S2WAT-main"
set "PROJECT=G:\GitHub\Latent_Style\Cycle-NCE"
set "TRAIN_PY=%ROOT%\train.py"
set "CONTENT_DIR=%ROOT%\input\Train\Content"
set "STYLE_DIR=%ROOT%\input\Train\Style"
set "VGG_PATH=%ROOT%\pre_trained_models\vgg_normalised.pth"
set "CKPT_DIR=%ROOT%\pre_trained_models\checkpoint_bs1_safe224"

if "%~1"=="" (
  set "CHUNK_EPOCHS=200"
) else (
  set "CHUNK_EPOCHS=%~1"
)

if "%~2"=="" (
  set "ROUNDS=1"
) else (
  set "ROUNDS=%~2"
)

if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"
set "KMP_AFFINITY=disabled"

echo [INFO] SAFE224 mode
echo [INFO] chunk epochs: %CHUNK_EPOCHS%
echo [INFO] rounds: %ROUNDS%
echo [INFO] checkpoint dir: %CKPT_DIR%

for /L %%R in (1,1,%ROUNDS%) do (
  echo.
  echo [INFO] ===== Round %%R / %ROUNDS% =====
  set "LATEST_CKPT="
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$d='%CKPT_DIR%'; if(Test-Path $d){$f=Get-ChildItem $d -File -Filter 'checkpoint_*_epoch.pkl' | Sort-Object { [int](($_.BaseName -replace 'checkpoint_','') -replace '_epoch','') } -Descending | Select-Object -First 1; if($f){$f.FullName}}"`) do (
    set "LATEST_CKPT=%%I"
  )

  if defined LATEST_CKPT (
    echo [INFO] resume from: !LATEST_CKPT!
    uv run --project "%PROJECT%" python "%TRAIN_PY%" ^
      --content_dir "%CONTENT_DIR%" ^
      --style_dir "%STYLE_DIR%" ^
      --vgg_dir "%VGG_PATH%" ^
      --batch_size 1 ^
      --img_size 224 ^
      --train_size 224 ^
      --precision amp ^
      --epoch %CHUNK_EPOCHS% ^
      --loss_count_interval 20 ^
      --checkpoint_save_interval 50 ^
      --checkpoint_save_path "%CKPT_DIR%" ^
      --resume_train True ^
      --checkpoint_import_path "!LATEST_CKPT!"
  ) else (
    echo [INFO] start from scratch
    uv run --project "%PROJECT%" python "%TRAIN_PY%" ^
      --content_dir "%CONTENT_DIR%" ^
      --style_dir "%STYLE_DIR%" ^
      --vgg_dir "%VGG_PATH%" ^
      --batch_size 1 ^
      --img_size 224 ^
      --train_size 224 ^
      --precision amp ^
      --epoch %CHUNK_EPOCHS% ^
      --loss_count_interval 20 ^
      --checkpoint_save_interval 50 ^
      --checkpoint_save_path "%CKPT_DIR%"
  )

  if errorlevel 1 (
    echo [ERROR] training failed in round %%R
    exit /b 1
  )
)

echo.
echo [INFO] all rounds completed
exit /b 0

