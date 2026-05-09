@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\..\..\.."

set "STATUS_LOG=experiments\active\full_dimensional_orthogonal_sweep_20\eval_status.csv"
echo name,eval_status,eval_rc>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo [g0_golden_pedestal] eval
python run_evaluation.py ".\exp\runs\fd20_g0_golden_pedestal" --output ".\exp\runs\fd20_g0_golden_pedestal\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g0_golden_pedestal,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g1_micro_only] eval
python run_evaluation.py ".\exp\runs\fd20_g1_micro_only" --output ".\exp\runs\fd20_g1_micro_only\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g1_micro_only,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g2_macro_only] eval
python run_evaluation.py ".\exp\runs\fd20_g2_macro_only" --output ".\exp\runs\fd20_g2_macro_only\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g2_macro_only,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g3_bimodal_split] eval
python run_evaluation.py ".\exp\runs\fd20_g3_bimodal_split" --output ".\exp\runs\fd20_g3_bimodal_split\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g3_bimodal_split,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g4_high_tension] eval
python run_evaluation.py ".\exp\runs\fd20_g4_high_tension" --output ".\exp\runs\fd20_g4_high_tension\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g4_high_tension,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g5_low_tension] eval
python run_evaluation.py ".\exp\runs\fd20_g5_low_tension" --output ".\exp\runs\fd20_g5_low_tension\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g5_low_tension,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g6_zero_friction] eval
python run_evaluation.py ".\exp\runs\fd20_g6_zero_friction" --output ".\exp\runs\fd20_g6_zero_friction\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g6_zero_friction,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g7_sharp_ot] eval
python run_evaluation.py ".\exp\runs\fd20_g7_sharp_ot" --output ".\exp\runs\fd20_g7_sharp_ot\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g7_sharp_ot,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g8_soft_ot] eval
python run_evaluation.py ".\exp\runs\fd20_g8_soft_ot" --output ".\exp\runs\fd20_g8_soft_ot\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g8_soft_ot,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g9_strict_l1] eval
python run_evaluation.py ".\exp\runs\fd20_g9_strict_l1" --output ".\exp\runs\fd20_g9_strict_l1\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g9_strict_l1,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g10_loose_l1] eval
python run_evaluation.py ".\exp\runs\fd20_g10_loose_l1" --output ".\exp\runs\fd20_g10_loose_l1\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g10_loose_l1,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g11_cycle_drop] eval
python run_evaluation.py ".\exp\runs\fd20_g11_cycle_drop" --output ".\exp\runs\fd20_g11_cycle_drop\full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g11_cycle_drop,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo Full-dimensional orthogonal sweep evaluation finished.
echo Status log: %STATUS_LOG%
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
