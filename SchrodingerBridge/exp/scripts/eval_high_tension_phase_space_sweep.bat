@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments\high_tension_phase_space_sweep\full_eval" mkdir "experiments\high_tension_phase_space_sweep\full_eval"
set "STATUS_LOG=experiments\high_tension_phase_space_sweep\eval_status.csv"
echo name,eval_status,eval_rc>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo Evaluating high-tension phase space sweep...
echo.

echo [g1_high_tension_base] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g1_high_tension_base" --output "experiments\high_tension_phase_space_sweep\full_eval\g1_high_tension_base" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g1_high_tension_base,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g2_swd_nuke] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g2_swd_nuke" --output "experiments\high_tension_phase_space_sweep\full_eval\g2_swd_nuke" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g2_swd_nuke,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g3_kinetic_vise] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g3_kinetic_vise" --output "experiments\high_tension_phase_space_sweep\full_eval\g3_kinetic_vise" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g3_kinetic_vise,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g4_brittle_flow] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g4_brittle_flow" --output "experiments\high_tension_phase_space_sweep\full_eval\g4_brittle_flow" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g4_brittle_flow,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g5_the_singularity] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g5_the_singularity" --output "experiments\high_tension_phase_space_sweep\full_eval\g5_the_singularity" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g5_the_singularity,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g6_cycle_ablation] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g6_cycle_ablation" --output "experiments\high_tension_phase_space_sweep\full_eval\g6_cycle_ablation" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g6_cycle_ablation,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g7_freq_ablation] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g7_freq_ablation" --output "experiments\high_tension_phase_space_sweep\full_eval\g7_freq_ablation" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g7_freq_ablation,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g8_sweet_spot] eval
python run_evaluation.py "experiments\high_tension_phase_space_sweep\g8_sweet_spot" --output "experiments\high_tension_phase_space_sweep\full_eval\g8_sweet_spot" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g8_sweet_spot,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo.
echo High-tension phase space sweep eval finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
