@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONUNBUFFERED=1
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_bodytransport_lowfree_fixed_w34_guard,ema_transport_adain_w34_guard,ema_transport_adain_w40_style --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend\ema_transport_moment > exp\vae_backend\ema_transport_moment_full.log 2>&1
