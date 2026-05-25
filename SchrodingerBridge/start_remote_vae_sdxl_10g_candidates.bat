@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_t01_recover,sdxl_style_push,sdxl_content_guard,sdxl_t01_fullish --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_10g_candidates
