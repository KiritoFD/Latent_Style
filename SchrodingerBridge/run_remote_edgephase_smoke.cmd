@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B tools\experiments\run_vae_backend_256_probe.py --variants ema_edgephase_w32_flatguard,ema_edgephase_w40_styleguard --epochs 1 --eval-epochs none --out-root exp\vae_backend\ema_edgephase_smoke --skip-existing-latents --max-train-batches 30
