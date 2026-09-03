@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "high_tension_phase_space_sweep" mkdir "high_tension_phase_space_sweep"
set "STATUS_LOG=high_tension_phase_space_sweep\train_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0

echo Running high-tension phase space sweep...
echo.

echo [g1_high_tension_base] train
python run.py --config "high_tension_phase_space_sweep\g1_high_tension_base.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g1_high_tension_base\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g1_high_tension_base,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g2_swd_nuke] train
python run.py --config "high_tension_phase_space_sweep\g2_swd_nuke.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g2_swd_nuke\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g2_swd_nuke,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g3_kinetic_vise] train
python run.py --config "high_tension_phase_space_sweep\g3_kinetic_vise.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g3_kinetic_vise\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g3_kinetic_vise,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g4_brittle_flow] train
python run.py --config "high_tension_phase_space_sweep\g4_brittle_flow.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g4_brittle_flow\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g4_brittle_flow,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g5_the_singularity] train
python run.py --config "high_tension_phase_space_sweep\g5_the_singularity.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g5_the_singularity\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g5_the_singularity,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g6_cycle_ablation] train
python run.py --config "high_tension_phase_space_sweep\g6_cycle_ablation.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g6_cycle_ablation\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g6_cycle_ablation,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g7_freq_ablation] train
python run.py --config "high_tension_phase_space_sweep\g7_freq_ablation.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g7_freq_ablation\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g7_freq_ablation,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g8_sweet_spot] train
python run.py --config "high_tension_phase_space_sweep\g8_sweet_spot.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "high_tension_phase_space_sweep\g8_sweet_spot\epoch_0080.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo g8_sweet_spot,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo High-tension phase space sweep finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
