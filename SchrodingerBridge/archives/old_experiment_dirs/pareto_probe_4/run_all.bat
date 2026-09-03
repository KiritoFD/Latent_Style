@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
set "PYTHON_EXE=C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHONHOME="

set "STATUS_LOG=pareto_probe_4\train_eval_status.csv"
echo name,train_status,train_rc,checkpoint_epoch_0020,eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo [S-add__K-3_C-2_W-10_Col-15] train
"%PYTHON_EXE%" run.py --config "pareto_probe_4/S-add__K-3_C-2_W-10_Col-15.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist "./exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
  set "EVAL_STATUS=SKIPPED"
  set "EVAL_RC=NA"
  set "BATCH_STATUS=NO"
) else (
  set "TRAIN_STATUS=OK"
  echo [S-add__K-3_C-2_W-10_Col-15] eval
  "%PYTHON_EXE%" run_evaluation.py "./exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15" --output "./exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if exist "./exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
)
echo S-add__K-3_C-2_W-10_Col-15,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-3_C-2_W-10_Col-15] train
"%PYTHON_EXE%" run.py --config "pareto_probe_4/S-norm_K-3_C-2_W-10_Col-15.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist "./exp/pareto_probe_4/S-norm_K-3_C-2_W-10_Col-15\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
  set "EVAL_STATUS=SKIPPED"
  set "EVAL_RC=NA"
  set "BATCH_STATUS=NO"
) else (
  set "TRAIN_STATUS=OK"
  echo [S-norm_K-3_C-2_W-10_Col-15] eval
  "%PYTHON_EXE%" run_evaluation.py "./exp/pareto_probe_4/S-norm_K-3_C-2_W-10_Col-15" --output "./exp/pareto_probe_4/S-norm_K-3_C-2_W-10_Col-15/full_eval" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if exist "./exp/pareto_probe_4/S-norm_K-3_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
)
echo S-norm_K-3_C-2_W-10_Col-15,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-5_C-5_W-15_Col-15] train
"%PYTHON_EXE%" run.py --config "pareto_probe_4/S-add__K-5_C-5_W-15_Col-15.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist "./exp/pareto_probe_4/S-add__K-5_C-5_W-15_Col-15\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
  set "EVAL_STATUS=SKIPPED"
  set "EVAL_RC=NA"
  set "BATCH_STATUS=NO"
) else (
  set "TRAIN_STATUS=OK"
  echo [S-add__K-5_C-5_W-15_Col-15] eval
  "%PYTHON_EXE%" run_evaluation.py "./exp/pareto_probe_4/S-add__K-5_C-5_W-15_Col-15" --output "./exp/pareto_probe_4/S-add__K-5_C-5_W-15_Col-15/full_eval" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if exist "./exp/pareto_probe_4/S-add__K-5_C-5_W-15_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
)
echo S-add__K-5_C-5_W-15_Col-15,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-5_C-5_W-15_Col-15] train
"%PYTHON_EXE%" run.py --config "pareto_probe_4/S-norm_K-5_C-5_W-15_Col-15.json"
set "TRAIN_RC=!ERRORLEVEL!"
if exist "./exp/pareto_probe_4/S-norm_K-5_C-5_W-15_Col-15\epoch_0020.pt" (set "CKPT_STATUS=YES") else (set "CKPT_STATUS=NO")
if not "!TRAIN_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "TRAIN_STATUS=FAIL"
  set "EVAL_STATUS=SKIPPED"
  set "EVAL_RC=NA"
  set "BATCH_STATUS=NO"
) else (
  set "TRAIN_STATUS=OK"
  echo [S-norm_K-5_C-5_W-15_Col-15] eval
  "%PYTHON_EXE%" run_evaluation.py "./exp/pareto_probe_4/S-norm_K-5_C-5_W-15_Col-15" --output "./exp/pareto_probe_4/S-norm_K-5_C-5_W-15_Col-15/full_eval" --batch_size 2
  set "EVAL_RC=!ERRORLEVEL!"
  if exist "./exp/pareto_probe_4/S-norm_K-5_C-5_W-15_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
  if not "!EVAL_RC!"=="0" (
    set /a FAIL_COUNT+=1
    set "EVAL_STATUS=FAIL"
  ) else (
    set "EVAL_STATUS=OK"
  )
)
echo S-norm_K-5_C-5_W-15_Col-15,!TRAIN_STATUS!,!TRAIN_RC!,!CKPT_STATUS!,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo Training done. Failures: %FAIL_COUNT%
exit /b %FAIL_COUNT%
