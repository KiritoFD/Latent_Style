@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "STATUS_LOG=screening_grid_3epoch_108\eval_status.csv"
echo name,eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo [S-none_K-1_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-1_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-1_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-1_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-2_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-2_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-2_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-none_K-4_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-none_K-4_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-none_K-4_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-1_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-1_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-1_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-2_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-2_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-2_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-add__K-4_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-add__K-4_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-add__K-4_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-1_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-1_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-1_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-2_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-2_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-2_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-0_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-0_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-0_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-0_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-0_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-0_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-0_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-0_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-0_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-2_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-2_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-2_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-2_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-2_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-2_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-2_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-2_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-2_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-5_W-10_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-5_W-10_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-5_W-10_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-10_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-5_W-10_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-5_W-20_Col-0] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-0" --output "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-0/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-0/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-5_W-20_Col-0,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [S-norm_K-4_C-5_W-20_Col-15] eval
python run_evaluation.py "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-15" --output "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-15/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/grid_search_3epoch/S-norm_K-4_C-5_W-20_Col-15/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo S-norm_K-4_C-5_W-20_Col-15,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo Eval done. Failures: %FAIL_COUNT%
exit /b %FAIL_COUNT%
