@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments\full_eval" mkdir "experiments\full_eval"
set "STATUS_LOG=experiments\latent_anchor_stress_60_eval_status.csv"
echo name,eval_status,eval_rc>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo Evaluating latent-anchor stress test 60 matrix...
echo.

echo [latent60_01_naked] eval
python run_evaluation.py "experiments\latent60_01_naked" --output "experiments\full_eval\latent60_01_naked" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo latent60_01_naked,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [latent60_02_patch_nce] eval
python run_evaluation.py "experiments\latent60_02_patch_nce" --output "experiments\full_eval\latent60_02_patch_nce" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo latent60_02_patch_nce,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [latent60_03_low_freq] eval
python run_evaluation.py "experiments\latent60_03_low_freq" --output "experiments\full_eval\latent60_03_low_freq" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo latent60_03_low_freq,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [latent60_04_nce_low_freq] eval
python run_evaluation.py "experiments\latent60_04_nce_low_freq" --output "experiments\full_eval\latent60_04_nce_low_freq" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo latent60_04_nce_low_freq,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo.
echo Latent-anchor stress test 60 eval finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
