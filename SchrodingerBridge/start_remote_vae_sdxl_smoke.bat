@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_s0_minimal --epochs 1 --eval-epochs 1 --out-root exp\vae_backend_256_sdxl_smoke
