@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl,flux1,flux2 --epochs 8 --eval-epochs 6,7,8 > exp\vae_backend_256_probe\runner_stdout.log 2>&1
