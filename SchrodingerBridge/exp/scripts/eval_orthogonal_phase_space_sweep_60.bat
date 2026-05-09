@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "orthogonal_phase_space_sweep_60\full_eval" mkdir "orthogonal_phase_space_sweep_60\full_eval"
set "STATUS_LOG=orthogonal_phase_space_sweep_60\eval_status.csv"
echo name,eval_status,eval_rc>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo Evaluating orthogonal phase space sweep 60...
echo.

echo [g0_universe_center] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g0_universe_center" --output "orthogonal_phase_space_sweep_60\full_eval\g0_universe_center" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g0_universe_center,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g1_absolute_release] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g1_absolute_release" --output "orthogonal_phase_space_sweep_60\full_eval\g1_absolute_release" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g1_absolute_release,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g2_absolute_freeze] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g2_absolute_freeze" --output "orthogonal_phase_space_sweep_60\full_eval\g2_absolute_freeze" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g2_absolute_freeze,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g3_gravity_black_hole] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g3_gravity_black_hole" --output "orthogonal_phase_space_sweep_60\full_eval\g3_gravity_black_hole" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g3_gravity_black_hole,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g4_gravity_vacuum] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g4_gravity_vacuum" --output "orthogonal_phase_space_sweep_60\full_eval\g4_gravity_vacuum" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g4_gravity_vacuum,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g5_midfreq_strangulation] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g5_midfreq_strangulation" --output "orthogonal_phase_space_sweep_60\full_eval\g5_midfreq_strangulation" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g5_midfreq_strangulation,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g6_structure_amnesty] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g6_structure_amnesty" --output "orthogonal_phase_space_sweep_60\full_eval\g6_structure_amnesty" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g6_structure_amnesty,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g7_flesh_stripping] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g7_flesh_stripping" --output "orthogonal_phase_space_sweep_60\full_eval\g7_flesh_stripping" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g7_flesh_stripping,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g8_absolute_nailgun] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g8_absolute_nailgun" --output "orthogonal_phase_space_sweep_60\full_eval\g8_absolute_nailgun" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g8_absolute_nailgun,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g9_cryogenic_hard_match] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g9_cryogenic_hard_match" --output "orthogonal_phase_space_sweep_60\full_eval\g9_cryogenic_hard_match" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g9_cryogenic_hard_match,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g10_thermal_soft_collapse] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g10_thermal_soft_collapse" --output "orthogonal_phase_space_sweep_60\full_eval\g10_thermal_soft_collapse" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g10_thermal_soft_collapse,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g11_blind_men_slicing] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g11_blind_men_slicing" --output "orthogonal_phase_space_sweep_60\full_eval\g11_blind_men_slicing" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g11_blind_men_slicing,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo [g12_limit_approximation] eval
python run_evaluation.py "orthogonal_phase_space_sweep_60\g12_limit_approximation" --output "orthogonal_phase_space_sweep_60\full_eval\g12_limit_approximation" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo g12_limit_approximation,!EVAL_STATUS!,!EVAL_RC!>>"%STATUS_LOG%"
echo.
echo.
echo Orthogonal phase space sweep 60 eval finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
