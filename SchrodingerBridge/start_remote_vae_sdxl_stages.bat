@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_s0_minimal,sdxl_s0_minimal_diffeo,sdxl_s0_stability,sdxl_s1_light_swd,sdxl_s2_balanced,sdxl_s3_style_push --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl
