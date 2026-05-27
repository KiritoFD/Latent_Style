@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B tools\experiments\run_vae_backend_256_probe.py --variants ema_bodymod_w32_guard,ema_bodymod_w36_style --epochs 1 --eval-epochs none --max-train-batches 30 --skip-existing-latents --out-root exp\vae_backend\ema_bodymod_smoke
endlocal
