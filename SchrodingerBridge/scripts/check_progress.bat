@echo off
set EVAL_DIR=C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003
set LOG=C:\Users\Administrator\logs\pixel256_eval.log
echo === PNG COUNT ===
dir /b "%EVAL_DIR%\*.png" 2>nul | find /c /v ""
echo === LATEST 3 PNGs ===
dir /b /o-d "%EVAL_DIR%\*.png" 2>nul | findstr /n "^" | findstr "^[1-3]:"
echo === LOG TAIL ===
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 10"
echo === GPU ===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo === DONE ===
