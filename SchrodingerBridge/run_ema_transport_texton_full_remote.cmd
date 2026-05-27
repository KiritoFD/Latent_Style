@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_transport_texton_w34_guard,ema_transport_texton_w40_style --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend\ema_transport_texton > exp\vae_backend\ema_transport_texton_task.log 2>&1
