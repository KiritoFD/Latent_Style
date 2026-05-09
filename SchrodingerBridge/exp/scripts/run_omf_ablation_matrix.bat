@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "STATUS_DIR=experiments\omf_ablation_matrix"
if not exist "%STATUS_DIR%" mkdir "%STATUS_DIR%"
set "STATUS_LOG=%STATUS_DIR%\run_status.csv"
echo name,train_status,train_rc,eval_status,eval_rc,checkpoint_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0
set /a TRAIN_FAIL_COUNT=0
set /a EVAL_FAIL_COUNT=0
set /a SKIP_EVAL_COUNT=0

echo Running OMF ablation matrix...
echo.

echo [01_strict_anchor] train
python run.py --config "experiments\omf_ablation_matrix\configs\01_strict_anchor.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\01_strict_anchor\epoch_0100.pt" (
  echo [01_strict_anchor] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\01_strict_anchor\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\01_strict_anchor" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [01_strict_anchor] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 01_strict_anchor,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [02_balanced_omf] train
python run.py --config "experiments\omf_ablation_matrix\configs\02_balanced_omf.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\02_balanced_omf\epoch_0100.pt" (
  echo [02_balanced_omf] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\02_balanced_omf\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\02_balanced_omf" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [02_balanced_omf] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 02_balanced_omf,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [03_aggressive_style] train
python run.py --config "experiments\omf_ablation_matrix\configs\03_aggressive_style.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\03_aggressive_style\epoch_0100.pt" (
  echo [03_aggressive_style] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\03_aggressive_style\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\03_aggressive_style" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [03_aggressive_style] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 03_aggressive_style,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [04_arch_skip] train
python run.py --config "experiments\omf_ablation_matrix\configs\04_arch_skip.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\04_arch_skip\epoch_0100.pt" (
  echo [04_arch_skip] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\04_arch_skip\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\04_arch_skip" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [04_arch_skip] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 04_arch_skip,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [05_color_05] train
python run.py --config "experiments\omf_ablation_matrix\configs\05_color_05.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\05_color_05\epoch_0100.pt" (
  echo [05_color_05] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\05_color_05\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\05_color_05" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [05_color_05] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 05_color_05,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [06_high_color] train
python run.py --config "experiments\omf_ablation_matrix\configs\06_high_color.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\06_high_color\epoch_0100.pt" (
  echo [06_high_color] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\06_high_color\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\06_high_color" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [06_high_color] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 06_high_color,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [07_pure_physics] train
python run.py --config "experiments\omf_ablation_matrix\configs\07_pure_physics.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\07_pure_physics\epoch_0100.pt" (
  echo [07_pure_physics] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\07_pure_physics\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\07_pure_physics" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [07_pure_physics] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 07_pure_physics,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo [08_heavy_repulsive] train
python run.py --config "experiments\omf_ablation_matrix\configs\08_heavy_repulsive.json"
set "TRAIN_RC=!ERRORLEVEL!"
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set /a TRAIN_FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
) else (
  set "TRAIN_STATUS=OK"
)

if exist "experiments\omf_ablation_matrix\artifacts\08_heavy_repulsive\epoch_0100.pt" (
  echo [08_heavy_repulsive] eval
  python run_evaluation.py "experiments\omf_ablation_matrix\artifacts\08_heavy_repulsive\epoch_0100.pt" --output "experiments\omf_ablation_matrix\full_eval\08_heavy_repulsive" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set /a EVAL_FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
  set "CKPT_STATUS=YES"
) else (
  echo [08_heavy_repulsive] checkpoint missing, skip eval
  set /a FAIL_COUNT+=1
  set /a SKIP_EVAL_COUNT+=1
  set "EVAL_RC=NA"
  set "EVAL_STATUS=SKIP"
  set "CKPT_STATUS=NO"
)
echo 08_heavy_repulsive,!TRAIN_STATUS!,!TRAIN_RC!,!EVAL_STATUS!,!EVAL_RC!,!CKPT_STATUS!>>"%STATUS_LOG%"
echo.
echo.
echo OMF ablation runs finished.
echo Status log: %STATUS_LOG%
echo Total failures: !FAIL_COUNT! ^| train: !TRAIN_FAIL_COUNT! ^| eval: !EVAL_FAIL_COUNT! ^| skipped eval: !SKIP_EVAL_COUNT!
if not "!FAIL_COUNT!"=="0" exit /b 1
exit /b 0
