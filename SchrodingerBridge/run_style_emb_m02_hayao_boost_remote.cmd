@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONUNBUFFERED=1
set OUT_ROOT=exp\style_embedding_mainline_calibration\ema_transport_adain_w34_e6_m02_hayao_boost
if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"

"C:\Program Files\Python312\python.exe" tools\experiments\run_style_embedding_mainline_calibration.py ^
  --checkpoint "exp\vae_backend\ema_transport_moment\ema_transport_adain_w34_guard\epoch_0006.pt" ^
  --latent-root "I:\Github\Latent_Style\latent-256-sd15-ema" ^
  --out-root "%OUT_ROOT%" ^
  --init-style-adapter "exp\style_embedding_mainline_calibration\ema_transport_adain_w34_e6_fulltrain\m02_embspatial_highpass_style\style_adapter.pt" ^
  --recipes "m03_m02_styleboost_balanced,m04_m02_styleboost_loose,m05_m02_midcolor_push" ^
  --target-style-ids "1" ^
  --eval-batch-size 8 ^
  --vae-model auto ^
  > "%OUT_ROOT%\run.log" 2>&1

echo exit=%ERRORLEVEL% >> "%OUT_ROOT%\run.log"
endlocal
