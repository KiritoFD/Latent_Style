@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
set "PYTHON_EXE=C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHONHOME="

set "STATUS_LOG=pareto_probe_4\eval_status.csv"
echo name,eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0

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
echo S-add__K-3_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

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
echo S-norm_K-3_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

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
echo S-add__K-5_C-5_W-15_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

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
echo S-norm_K-5_C-5_W-15_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo Eval done. Failures: %FAIL_COUNT%
exit /b %FAIL_COUNT%
