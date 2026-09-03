@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants ema_sconv_hp_w28_guard,ema_sconv_hp_w36_style --epochs 8 --eval-epochs 6,7,8 --out-root exp\vae_backend\ema_sconv_highpass --skip-existing-latents > exp\vae_backend\ema_sconv_highpass\runner_stdout.log 2>&1
