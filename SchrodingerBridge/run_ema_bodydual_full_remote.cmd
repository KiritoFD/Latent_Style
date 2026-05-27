@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py -3 -B tools\experiments\run_vae_backend_256_probe.py --variants ema_bodydual_w40_style --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend\ema_bodydual
endlocal
