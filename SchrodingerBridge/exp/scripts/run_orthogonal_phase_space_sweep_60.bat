@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "orthogonal_phase_space_sweep_60" mkdir "orthogonal_phase_space_sweep_60"
set "STATUS_LOG=orthogonal_phase_space_sweep_60\train_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0

echo Running orthogonal phase space sweep 60...
echo.

echo [g0_universe_center] train
python run.py --config "orthogonal_phase_space_sweep_60\g0_universe_center.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g0_universe_center\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g0_universe_center,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g1_absolute_release] train
python run.py --config "orthogonal_phase_space_sweep_60\g1_absolute_release.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g1_absolute_release\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g1_absolute_release,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g2_absolute_freeze] train
python run.py --config "orthogonal_phase_space_sweep_60\g2_absolute_freeze.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g2_absolute_freeze\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g2_absolute_freeze,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g3_gravity_black_hole] train
python run.py --config "orthogonal_phase_space_sweep_60\g3_gravity_black_hole.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g3_gravity_black_hole\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g3_gravity_black_hole,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g4_gravity_vacuum] train
python run.py --config "orthogonal_phase_space_sweep_60\g4_gravity_vacuum.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g4_gravity_vacuum\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g4_gravity_vacuum,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g5_midfreq_strangulation] train
python run.py --config "orthogonal_phase_space_sweep_60\g5_midfreq_strangulation.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g5_midfreq_strangulation\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g5_midfreq_strangulation,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g6_structure_amnesty] train
python run.py --config "orthogonal_phase_space_sweep_60\g6_structure_amnesty.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g6_structure_amnesty\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g6_structure_amnesty,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g7_flesh_stripping] train
python run.py --config "orthogonal_phase_space_sweep_60\g7_flesh_stripping.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g7_flesh_stripping\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g7_flesh_stripping,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g8_absolute_nailgun] train
python run.py --config "orthogonal_phase_space_sweep_60\g8_absolute_nailgun.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g8_absolute_nailgun\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g8_absolute_nailgun,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g9_cryogenic_hard_match] train
python run.py --config "orthogonal_phase_space_sweep_60\g9_cryogenic_hard_match.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g9_cryogenic_hard_match\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g9_cryogenic_hard_match,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g10_thermal_soft_collapse] train
python run.py --config "orthogonal_phase_space_sweep_60\g10_thermal_soft_collapse.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g10_thermal_soft_collapse\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g10_thermal_soft_collapse,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g11_blind_men_slicing] train
python run.py --config "orthogonal_phase_space_sweep_60\g11_blind_men_slicing.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g11_blind_men_slicing\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g11_blind_men_slicing,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g12_limit_approximation] train
python run.py --config "orthogonal_phase_space_sweep_60\g12_limit_approximation.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "orthogonal_phase_space_sweep_60\g12_limit_approximation\epoch_0060.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g12_limit_approximation,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo Orthogonal phase space sweep 60 finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
