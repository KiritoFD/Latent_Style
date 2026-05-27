@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
if not exist exp\vae_backend\ema_sconv_semantic mkdir exp\vae_backend\ema_sconv_semantic
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_sconv_semantic_w34_guard,ema_sconv_semantic_w44_style --epochs 8 --eval-epochs 6,7,8 --out-root exp\vae_backend\ema_sconv_semantic --skip-existing-latents > exp\vae_backend\ema_sconv_semantic\runner_stdout.log 2>&1
