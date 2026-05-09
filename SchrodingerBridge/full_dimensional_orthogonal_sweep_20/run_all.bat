@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "STATUS_LOG=full_dimensional_orthogonal_sweep_20\train_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo [g0_golden_pedestal] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g0_golden_pedestal.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g0_golden_pedestal\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g0_golden_pedestal,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g1_micro_only] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g1_micro_only.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g1_micro_only\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g1_micro_only,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g2_macro_only] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g2_macro_only.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g2_macro_only\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g2_macro_only,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g3_bimodal_split] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g3_bimodal_split.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g3_bimodal_split\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g3_bimodal_split,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g4_high_tension] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g4_high_tension.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g4_high_tension\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g4_high_tension,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g5_low_tension] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g5_low_tension.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g5_low_tension\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g5_low_tension,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g6_zero_friction] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g6_zero_friction.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g6_zero_friction\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g6_zero_friction,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g7_sharp_ot] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g7_sharp_ot.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g7_sharp_ot\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g7_sharp_ot,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g8_soft_ot] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g8_soft_ot.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g8_soft_ot\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g8_soft_ot,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g9_strict_l1] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g9_strict_l1.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g9_strict_l1\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g9_strict_l1,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g10_loose_l1] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g10_loose_l1.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g10_loose_l1\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g10_loose_l1,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [g11_cycle_drop] train
python run.py --config "full_dimensional_orthogonal_sweep_20\g11_cycle_drop.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist ".\exp\runs\fd20_g11_cycle_drop\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)
echo g11_cycle_drop,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo Full-dimensional orthogonal sweep training finished.
echo Status log: %STATUS_LOG%
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
