@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_mem_b96,sdxl_mem_b128,sdxl_mem_b160,sdxl_mem_b192 --epochs 1 --eval-epochs none --max-train-batches 30 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_10g_ladder
