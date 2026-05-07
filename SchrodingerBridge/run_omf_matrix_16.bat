@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist "experiments" mkdir "experiments"
set "STATUS_LOG=experiments\omf_matrix_16_run_status.csv"
echo name,train_status,train_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0

echo Running OMF 16-run matrix...
echo.

echo [01_omf_swd_15] train
python run.py --config "experiments\01_omf_swd_15.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\01_omf_swd_15\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 01_omf_swd_15,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [02_omf_swd_30] train
python run.py --config "experiments\02_omf_swd_30.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\02_omf_swd_30\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 02_omf_swd_30,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [03_omf_swd_45] train
python run.py --config "experiments\03_omf_swd_45.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\03_omf_swd_45\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 03_omf_swd_45,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [04_anchor_kin_only] train
python run.py --config "experiments\04_anchor_kin_only.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\04_anchor_kin_only\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 04_anchor_kin_only,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [05_anchor_ot_mse_only] train
python run.py --config "experiments\05_anchor_ot_mse_only.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\05_anchor_ot_mse_only\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 05_anchor_ot_mse_only,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [06_anchor_skip_only] train
python run.py --config "experiments\06_anchor_skip_only.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\06_anchor_skip_only\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 06_anchor_skip_only,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [07_anchor_hybrid_all] train
python run.py --config "experiments\07_anchor_hybrid_all.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\07_anchor_hybrid_all\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 07_anchor_hybrid_all,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [08_color_00] train
python run.py --config "experiments\08_color_00.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\08_color_00\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 08_color_00,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [09_color_25] train
python run.py --config "experiments\09_color_25.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\09_color_25\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 09_color_25,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [10_color_50] train
python run.py --config "experiments\10_color_50.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\10_color_50\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 10_color_50,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [11_repel_00] train
python run.py --config "experiments\11_repel_00.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\11_repel_00\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 11_repel_00,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [12_repel_05] train
python run.py --config "experiments\12_repel_05.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\12_repel_05\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 12_repel_05,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [13_repel_10] train
python run.py --config "experiments\13_repel_10.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\13_repel_10\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 13_repel_10,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [14_extreme_free_flow] train
python run.py --config "experiments\14_extreme_free_flow.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\14_extreme_free_flow\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 14_extreme_free_flow,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [15_extreme_stiff_ode] train
python run.py --config "experiments\15_extreme_stiff_ode.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\15_extreme_stiff_ode\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 15_extreme_stiff_ode,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [16_the_god_weight] train
python run.py --config "experiments\16_the_god_weight.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\16_the_god_weight\epoch_0160.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
echo 16_the_god_weight,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo OMF 16-run training matrix finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
