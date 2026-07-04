@echo off
echo === REMOTE REPO ===
dir /b "C:\Users\Administrator\SchrodingerBridge" 2>nul
echo === CONFIG FILE ===
if exist "C:\Users\Administrator\configs\630_latent_256.json" (echo EXISTS) else (echo MISSING)
echo === EVAL SCRIPT ===
if exist "C:\Users\Administrator\scripts\eval_pixel128.py" (echo eval_pixel128.py EXISTS) else (echo eval_pixel128.py MISSING)
echo === RUN.PY ===
if exist "C:\Users\Administrator\run.py" (echo run.py EXISTS) else (echo run.py MISSING)
echo === GPU ===
nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader
echo === DONE ===
