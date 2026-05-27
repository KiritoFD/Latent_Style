@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
"C:\Program Files\Python312\python.exe" tools\experiments\run_vae_backend_256_probe.py --variants ema_hardcontent_w18_anchor,ema_hardcontent_w24_anchor,ema_amp_only_w24_anchor,ema_identity_w24_anchor,ema_plain4_spectral_iso_w32 --epochs 8 --eval-epochs 6,7,8 --out-root exp\vae_backend_256_ema_arch_adapt_barrier --skip-existing-latents > exp\vae_backend_256_ema_arch_adapt_barrier_launcher.log 2>&1
