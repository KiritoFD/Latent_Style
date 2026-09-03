@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments" mkdir "experiments"
set "STATUS_LOG=experiments\arch_stress_test_60_run_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0

echo Running arch stress test 60 matrix...
echo.

echo [arch60_00_baseline_naked] train
python run.py --config "experiments\arch60_00_baseline_naked.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\arch60_00_baseline_naked\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo arch60_00_baseline_naked,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [arch60_01_arch_add_proj] train
python run.py --config "experiments\arch60_01_arch_add_proj.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\arch60_01_arch_add_proj\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo arch60_01_arch_add_proj,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [arch60_02_math_ot_mse] train
python run.py --config "experiments\arch60_02_math_ot_mse.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\arch60_02_math_ot_mse\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo arch60_02_math_ot_mse,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [arch60_03_feat_retention] train
python run.py --config "experiments\arch60_03_feat_retention.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\arch60_03_feat_retention\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo arch60_03_feat_retention,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [arch60_04_ultimate_armor] train
python run.py --config "experiments\arch60_04_ultimate_armor.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\arch60_04_ultimate_armor\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo arch60_04_ultimate_armor,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo Arch stress test 60 finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
