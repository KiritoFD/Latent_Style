@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments\full_eval" mkdir "experiments\full_eval"
set "STATUS_LOG=experiments\arch_stress_test_60_eval_status.csv"
echo name,eval_status,eval_rc>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo Evaluating arch stress test 60 matrix...
echo.

echo [arch60_00_baseline_naked] eval
python run_evaluation.py "experiments\arch60_00_baseline_naked" --output "experiments\full_eval\arch60_00_baseline_naked" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo arch60_00_baseline_naked,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [arch60_01_arch_add_proj] eval
python run_evaluation.py "experiments\arch60_01_arch_add_proj" --output "experiments\full_eval\arch60_01_arch_add_proj" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo arch60_01_arch_add_proj,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [arch60_02_math_ot_mse] eval
python run_evaluation.py "experiments\arch60_02_math_ot_mse" --output "experiments\full_eval\arch60_02_math_ot_mse" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo arch60_02_math_ot_mse,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [arch60_03_feat_retention] eval
python run_evaluation.py "experiments\arch60_03_feat_retention" --output "experiments\full_eval\arch60_03_feat_retention" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo arch60_03_feat_retention,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [arch60_04_ultimate_armor] eval
python run_evaluation.py "experiments\arch60_04_ultimate_armor" --output "experiments\full_eval\arch60_04_ultimate_armor" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo arch60_04_ultimate_armor,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo.
echo Arch stress test 60 eval finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
