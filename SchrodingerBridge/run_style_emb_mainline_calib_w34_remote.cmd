@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONUNBUFFERED=1
set OUT_ROOT=exp\style_embedding_mainline_calibration\ema_transport_adain_w34_e6_fulltrain
if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"

"C:\Program Files\Python312\python.exe" tools\experiments\run_style_embedding_mainline_calibration.py ^
  --checkpoint "exp\vae_backend\ema_transport_moment\ema_transport_adain_w34_guard\epoch_0006.pt" ^
  --latent-root "I:\Github\Latent_Style\latent-256-sd15-ema" ^
  --out-root "%OUT_ROOT%" ^
  --recipes "m00_emb_swd_anchor,m01_embspatial_swd_anchor,m02_embspatial_highpass_style" ^
  --target-style-ids "1,2,3,4" ^
  --eval-batch-size 8 ^
  --vae-model auto ^
  > "%OUT_ROOT%\run.log" 2>&1

echo exit=%ERRORLEVEL% >> "%OUT_ROOT%\run.log"
endlocal
