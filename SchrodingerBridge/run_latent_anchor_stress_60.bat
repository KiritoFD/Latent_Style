@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments" mkdir "experiments"
set "STATUS_LOG=experiments\latent_anchor_stress_60_run_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0

echo Running latent-anchor stress test 60 matrix...
echo.

echo [latent60_01_naked] train
python run.py --config "experiments\latent60_01_naked.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\latent60_01_naked\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo latent60_01_naked,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [latent60_02_patch_nce] train
python run.py --config "experiments\latent60_02_patch_nce.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\latent60_02_patch_nce\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo latent60_02_patch_nce,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [latent60_03_low_freq] train
python run.py --config "experiments\latent60_03_low_freq.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\latent60_03_low_freq\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo latent60_03_low_freq,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [latent60_04_nce_low_freq] train
python run.py --config "experiments\latent60_04_nce_low_freq.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\latent60_04_nce_low_freq\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo latent60_04_nce_low_freq,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo Latent-anchor stress test 60 finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
