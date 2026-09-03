@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -B tools\experiments\run_vae_backend_256_probe.py --variants ema_static_sconv_w32_guard,ema_static_sconv_w40_style --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend\ema_static_sconv
endlocal
