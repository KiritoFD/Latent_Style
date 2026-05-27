@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
if not exist exp\vae_backend\ema_sconv_support mkdir exp\vae_backend\ema_sconv_support
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_sconv_support_w30_guard,ema_sconv_support_w40_style --epochs 8 --eval-epochs 6,7,8 --out-root exp\vae_backend\ema_sconv_support --skip-existing-latents > exp\vae_backend\ema_sconv_support\runner_stdout.log 2>&1
