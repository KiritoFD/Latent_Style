@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B tools\experiments\run_vae_backend_256_probe.py --variants ema_bodyblend_w28_guard,ema_skip_adaptive_w28_guard,ema_sinkhorn_body_w28_guard --epochs 1 --eval-epochs none --max-train-batches 30 --skip-existing-latents --out-root exp\vae_backend\ema_bodyroute_smoke
endlocal
