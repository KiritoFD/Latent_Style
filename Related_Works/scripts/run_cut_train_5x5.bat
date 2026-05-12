@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem =========================
rem CUT 5x5 serial training
rem =========================
rem One target style per run, strictly sequential.
rem No parallel jobs.

set "PY=G:\GitHub\Latent_Style\Related_Works\envs\cut\Scripts\python.exe"
set "CUT_REPO=G:\GitHub\Latent_Style\Related_Works\external\CUT"
set "DATASETS_ROOT=G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\datasets"
set "CKPT_ROOT=G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\checkpoints"
set "LOG_ROOT=G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\logs"

set "N_EPOCHS=20"
set "N_EPOCHS_DECAY=0"
set "BATCH_SIZE=6"
set "NUM_THREADS=4"
set "LOAD_SIZE=256"
set "CROP_SIZE=256"
set "NUM_PATCHES=160"
set "NCE_IDT=False"
set "NCE_LAYERS=0,4,8"
set "NETG=resnet_6blocks"
set "NGF=48"
set "NDF=48"

if not exist "%PY%" (
  echo [ERR] Python not found: %PY%
  exit /b 1
)

if not exist "%CUT_REPO%\train.py" (
  echo [ERR] CUT repo not found or invalid: %CUT_REPO%
  exit /b 1
)

if not exist "%DATASETS_ROOT%" (
  echo [ERR] Dataset root not found: %DATASETS_ROOT%
  exit /b 1
)

if not exist "%CKPT_ROOT%" mkdir "%CKPT_ROOT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"

pushd "%CUT_REPO%"

set "TARGETS=cezanne Hayao monet photo vangogh"

for %%T in (%TARGETS%) do (
  set "NAME=cut_to_%%T"
  set "DATAROOT=%DATASETS_ROOT%\to_%%T"
  set "LOG=%LOG_ROOT%\train_!NAME!.log"

  if not exist "!DATAROOT!\trainA" (
    echo [ERR] Missing dataset trainA: !DATAROOT!\trainA
    popd
    exit /b 1
  )
  if not exist "!DATAROOT!\trainB" (
    echo [ERR] Missing dataset trainB: !DATAROOT!\trainB
    popd
    exit /b 1
  )

  echo [CUT] START !NAME!
  echo [CUT] log: !LOG!

  "%PY%" train.py ^
    --dataroot "!DATAROOT!" ^
    --name "!NAME!" ^
    --CUT_mode CUT ^
    --nce_idt %NCE_IDT% ^
    --nce_layers %NCE_LAYERS% ^
    --n_epochs %N_EPOCHS% ^
    --n_epochs_decay %N_EPOCHS_DECAY% ^
    --batch_size %BATCH_SIZE% ^
    --num_threads %NUM_THREADS% ^
    --netG %NETG% ^
    --ngf %NGF% ^
    --ndf %NDF% ^
    --load_size %LOAD_SIZE% ^
    --crop_size %CROP_SIZE% ^
    --num_patches %NUM_PATCHES% ^
    --preprocess resize_and_crop ^
    --display_id -1 ^
    --save_epoch_freq 10 ^
    --checkpoints_dir "%CKPT_ROOT%" ^
    >> "!LOG!" 2>&1

  if errorlevel 1 (
    echo [ERR] FAILED !NAME! ^(see log: !LOG!^)
    popd
    exit /b 1
  )

  echo [CUT] DONE !NAME!
)

popd
echo [CUT] ALL DONE
exit /b 0
