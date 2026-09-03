@echo off
setlocal
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "STATUS_LOG=next_round_80\eval_status.csv"
echo name,eval_status,eval_rc,batch_summary_exists>"%STATUS_LOG%"
set /a FAIL_COUNT=0

echo [E001] eval
python run_evaluation.py "./exp/next_round_80/E001" --output "./exp/next_round_80/E001/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E001/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E001,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E002] eval
python run_evaluation.py "./exp/next_round_80/E002" --output "./exp/next_round_80/E002/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E002/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E002,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E003] eval
python run_evaluation.py "./exp/next_round_80/E003" --output "./exp/next_round_80/E003/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E003/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E003,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E004] eval
python run_evaluation.py "./exp/next_round_80/E004" --output "./exp/next_round_80/E004/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E004/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E004,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E005] eval
python run_evaluation.py "./exp/next_round_80/E005" --output "./exp/next_round_80/E005/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E005/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E005,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E006] eval
python run_evaluation.py "./exp/next_round_80/E006" --output "./exp/next_round_80/E006/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E006/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E006,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E007] eval
python run_evaluation.py "./exp/next_round_80/E007" --output "./exp/next_round_80/E007/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E007/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E007,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E008] eval
python run_evaluation.py "./exp/next_round_80/E008" --output "./exp/next_round_80/E008/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E008/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E008,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E009] eval
python run_evaluation.py "./exp/next_round_80/E009" --output "./exp/next_round_80/E009/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E009/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E009,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E010] eval
python run_evaluation.py "./exp/next_round_80/E010" --output "./exp/next_round_80/E010/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E010/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E010,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E011] eval
python run_evaluation.py "./exp/next_round_80/E011" --output "./exp/next_round_80/E011/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E011/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E011,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E012] eval
python run_evaluation.py "./exp/next_round_80/E012" --output "./exp/next_round_80/E012/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E012/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E012,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E013] eval
python run_evaluation.py "./exp/next_round_80/E013" --output "./exp/next_round_80/E013/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E013/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E013,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E014] eval
python run_evaluation.py "./exp/next_round_80/E014" --output "./exp/next_round_80/E014/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E014/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E014,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E015] eval
python run_evaluation.py "./exp/next_round_80/E015" --output "./exp/next_round_80/E015/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E015/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E015,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E016] eval
python run_evaluation.py "./exp/next_round_80/E016" --output "./exp/next_round_80/E016/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E016/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E016,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E017] eval
python run_evaluation.py "./exp/next_round_80/E017" --output "./exp/next_round_80/E017/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E017/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E017,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E018] eval
python run_evaluation.py "./exp/next_round_80/E018" --output "./exp/next_round_80/E018/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E018/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E018,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E019] eval
python run_evaluation.py "./exp/next_round_80/E019" --output "./exp/next_round_80/E019/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E019/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E019,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E020] eval
python run_evaluation.py "./exp/next_round_80/E020" --output "./exp/next_round_80/E020/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E020/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E020,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E021] eval
python run_evaluation.py "./exp/next_round_80/E021" --output "./exp/next_round_80/E021/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E021/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E021,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E022] eval
python run_evaluation.py "./exp/next_round_80/E022" --output "./exp/next_round_80/E022/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E022/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E022,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E023] eval
python run_evaluation.py "./exp/next_round_80/E023" --output "./exp/next_round_80/E023/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E023/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E023,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E024] eval
python run_evaluation.py "./exp/next_round_80/E024" --output "./exp/next_round_80/E024/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E024/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E024,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E025] eval
python run_evaluation.py "./exp/next_round_80/E025" --output "./exp/next_round_80/E025/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E025/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E025,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E026] eval
python run_evaluation.py "./exp/next_round_80/E026" --output "./exp/next_round_80/E026/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E026/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E026,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E027] eval
python run_evaluation.py "./exp/next_round_80/E027" --output "./exp/next_round_80/E027/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E027/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E027,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E028] eval
python run_evaluation.py "./exp/next_round_80/E028" --output "./exp/next_round_80/E028/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E028/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E028,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E029] eval
python run_evaluation.py "./exp/next_round_80/E029" --output "./exp/next_round_80/E029/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E029/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E029,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E030] eval
python run_evaluation.py "./exp/next_round_80/E030" --output "./exp/next_round_80/E030/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E030/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E030,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E031] eval
python run_evaluation.py "./exp/next_round_80/E031" --output "./exp/next_round_80/E031/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E031/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E031,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E032] eval
python run_evaluation.py "./exp/next_round_80/E032" --output "./exp/next_round_80/E032/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E032/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E032,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E033] eval
python run_evaluation.py "./exp/next_round_80/E033" --output "./exp/next_round_80/E033/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E033/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E033,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E034] eval
python run_evaluation.py "./exp/next_round_80/E034" --output "./exp/next_round_80/E034/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E034/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E034,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E035] eval
python run_evaluation.py "./exp/next_round_80/E035" --output "./exp/next_round_80/E035/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E035/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E035,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E036] eval
python run_evaluation.py "./exp/next_round_80/E036" --output "./exp/next_round_80/E036/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E036/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E036,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E037] eval
python run_evaluation.py "./exp/next_round_80/E037" --output "./exp/next_round_80/E037/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E037/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E037,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E038] eval
python run_evaluation.py "./exp/next_round_80/E038" --output "./exp/next_round_80/E038/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E038/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E038,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E039] eval
python run_evaluation.py "./exp/next_round_80/E039" --output "./exp/next_round_80/E039/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E039/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E039,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E040] eval
python run_evaluation.py "./exp/next_round_80/E040" --output "./exp/next_round_80/E040/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E040/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E040,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E041] eval
python run_evaluation.py "./exp/next_round_80/E041" --output "./exp/next_round_80/E041/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E041/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E041,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E042] eval
python run_evaluation.py "./exp/next_round_80/E042" --output "./exp/next_round_80/E042/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E042/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E042,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E043] eval
python run_evaluation.py "./exp/next_round_80/E043" --output "./exp/next_round_80/E043/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E043/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E043,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E044] eval
python run_evaluation.py "./exp/next_round_80/E044" --output "./exp/next_round_80/E044/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E044/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E044,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E045] eval
python run_evaluation.py "./exp/next_round_80/E045" --output "./exp/next_round_80/E045/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E045/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E045,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E046] eval
python run_evaluation.py "./exp/next_round_80/E046" --output "./exp/next_round_80/E046/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E046/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E046,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E047] eval
python run_evaluation.py "./exp/next_round_80/E047" --output "./exp/next_round_80/E047/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E047/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E047,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E048] eval
python run_evaluation.py "./exp/next_round_80/E048" --output "./exp/next_round_80/E048/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E048/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E048,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E049] eval
python run_evaluation.py "./exp/next_round_80/E049" --output "./exp/next_round_80/E049/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E049/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E049,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E050] eval
python run_evaluation.py "./exp/next_round_80/E050" --output "./exp/next_round_80/E050/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E050/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E050,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E051] eval
python run_evaluation.py "./exp/next_round_80/E051" --output "./exp/next_round_80/E051/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E051/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E051,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E052] eval
python run_evaluation.py "./exp/next_round_80/E052" --output "./exp/next_round_80/E052/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E052/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E052,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E053] eval
python run_evaluation.py "./exp/next_round_80/E053" --output "./exp/next_round_80/E053/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E053/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E053,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E054] eval
python run_evaluation.py "./exp/next_round_80/E054" --output "./exp/next_round_80/E054/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E054/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E054,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E055] eval
python run_evaluation.py "./exp/next_round_80/E055" --output "./exp/next_round_80/E055/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E055/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E055,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E056] eval
python run_evaluation.py "./exp/next_round_80/E056" --output "./exp/next_round_80/E056/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E056/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E056,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E057] eval
python run_evaluation.py "./exp/next_round_80/E057" --output "./exp/next_round_80/E057/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E057/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E057,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E058] eval
python run_evaluation.py "./exp/next_round_80/E058" --output "./exp/next_round_80/E058/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E058/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E058,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E059] eval
python run_evaluation.py "./exp/next_round_80/E059" --output "./exp/next_round_80/E059/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E059/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E059,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E060] eval
python run_evaluation.py "./exp/next_round_80/E060" --output "./exp/next_round_80/E060/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E060/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E060,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E061] eval
python run_evaluation.py "./exp/next_round_80/E061" --output "./exp/next_round_80/E061/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E061/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E061,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E062] eval
python run_evaluation.py "./exp/next_round_80/E062" --output "./exp/next_round_80/E062/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E062/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E062,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E063] eval
python run_evaluation.py "./exp/next_round_80/E063" --output "./exp/next_round_80/E063/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E063/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E063,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E064] eval
python run_evaluation.py "./exp/next_round_80/E064" --output "./exp/next_round_80/E064/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E064/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E064,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E065] eval
python run_evaluation.py "./exp/next_round_80/E065" --output "./exp/next_round_80/E065/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E065/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E065,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E066] eval
python run_evaluation.py "./exp/next_round_80/E066" --output "./exp/next_round_80/E066/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E066/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E066,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E067] eval
python run_evaluation.py "./exp/next_round_80/E067" --output "./exp/next_round_80/E067/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E067/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E067,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E068] eval
python run_evaluation.py "./exp/next_round_80/E068" --output "./exp/next_round_80/E068/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E068/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E068,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E069] eval
python run_evaluation.py "./exp/next_round_80/E069" --output "./exp/next_round_80/E069/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E069/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E069,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E070] eval
python run_evaluation.py "./exp/next_round_80/E070" --output "./exp/next_round_80/E070/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E070/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E070,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E071] eval
python run_evaluation.py "./exp/next_round_80/E071" --output "./exp/next_round_80/E071/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E071/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E071,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E072] eval
python run_evaluation.py "./exp/next_round_80/E072" --output "./exp/next_round_80/E072/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E072/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E072,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E073] eval
python run_evaluation.py "./exp/next_round_80/E073" --output "./exp/next_round_80/E073/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E073/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E073,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E074] eval
python run_evaluation.py "./exp/next_round_80/E074" --output "./exp/next_round_80/E074/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E074/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E074,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E075] eval
python run_evaluation.py "./exp/next_round_80/E075" --output "./exp/next_round_80/E075/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E075/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E075,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E076] eval
python run_evaluation.py "./exp/next_round_80/E076" --output "./exp/next_round_80/E076/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E076/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E076,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E077] eval
python run_evaluation.py "./exp/next_round_80/E077" --output "./exp/next_round_80/E077/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E077/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E077,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E078] eval
python run_evaluation.py "./exp/next_round_80/E078" --output "./exp/next_round_80/E078/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E078/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E078,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E079] eval
python run_evaluation.py "./exp/next_round_80/E079" --output "./exp/next_round_80/E079/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E079/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E079,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo [E080] eval
python run_evaluation.py "./exp/next_round_80/E080" --output "./exp/next_round_80/E080/full_eval" --batch_size 2
set "EVAL_RC=!ERRORLEVEL!"
if exist "./exp/next_round_80/E080/full_eval\batch_summary.csv" (set "BATCH_STATUS=YES") else (set "BATCH_STATUS=NO")
if not "!EVAL_RC!"=="0" (
  set /a FAIL_COUNT+=1
  set "EVAL_STATUS=FAIL"
) else (
  set "EVAL_STATUS=OK"
)
echo E080,!EVAL_STATUS!,!EVAL_RC!,!BATCH_STATUS!>>"%STATUS_LOG%"
echo.

echo Eval done. Failures: %FAIL_COUNT%
exit /b %FAIL_COUNT%
