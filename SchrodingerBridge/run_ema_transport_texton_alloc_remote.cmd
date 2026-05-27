@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONUNBUFFERED=1
set OUT_ROOT=exp\vae_backend\ema_transport_texton_alloc
if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"

"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py ^
  --variants ema_transport_texton_alloc_w34,ema_transport_texton_alloc_hayao_w36,ema_transport_texton_alloc_cap_w34 ^
  --epochs 8 ^
  --eval-epochs 6,7,8 ^
  --skip-existing-latents ^
  --out-root "%OUT_ROOT%" ^
  > "%OUT_ROOT%\task.log" 2>&1

echo exit=%ERRORLEVEL% >> "%OUT_ROOT%\task.log"
endlocal
