@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_semantic_moment_adain_w30_guard,ema_semantic_moment_adain_w38_style --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend\ema_semantic_moment
