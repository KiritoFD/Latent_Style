@echo off
set EVAL_DIR=C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003
echo === PNG COUNT ===
dir /b "%EVAL_DIR%\*.png" 2>nul | find /c /v ""
echo === ALL FILES ===
dir /b "%EVAL_DIR%\" 2>nul
echo === LATEST PNGs (top 5) ===
dir /b /o-d "%EVAL_DIR%\*.png" 2>nul | findstr /n "^" | findstr "^[1-5]:"
echo === LOG TAIL ===
if exist C:\Users\Administrator\logs\pixel256_eval.log (
  powershell -Command "Get-Content C:\Users\Administrator\logs\pixel256_eval.log -Tail 20"
) else (
  echo NO_LOG_FILE
)
echo === GPU ===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo === DONE ===
