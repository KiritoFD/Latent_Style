@echo off
setlocal
setlocal EnableDelayedExpansion

cd /d "%~dp0"

python tools\setup_semantic_overfit_small.py
if errorlevel 1 exit /b 1

set "STATUS_LOG=experiments\semantic_overfit_small\run_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"

for %%N in (A_baseline B_no_kinetic C_no_low_freq) do (
  echo [%%N] train
  python run.py --config "experiments\semantic_overfit_small\configs\%%N.json"
  set "TRAIN_RC=!ERRORLEVEL!"
  if not "!TRAIN_RC!"=="0" (
    set "TRAIN_STATUS=FAIL"
  ) else (
    set "TRAIN_STATUS=OK"
  )

  if exist "experiments\semantic_overfit_small\%%N\epoch_0003.pt" (
    set "CKPT_STATUS=YES"
  ) else (
    set "CKPT_STATUS=NO"
  )

  echo %%N,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
  echo.
)

exit /b 0
