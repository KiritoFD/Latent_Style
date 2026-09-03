@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B tools\experiments\run_vae_backend_256_probe.py --variants ema_routed_w36_texton,ema_routed_w44_stylepush --epochs 8 --eval-epochs 6,7,8 --out-root exp\vae_backend\ema_routed --skip-existing-latents
