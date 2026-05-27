@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_t01max_w30_k05,sdxl_t01max_w40_k025,sdxl_t01max_allstyle,sdxl_t01max_allstyle_term8,sdxl_t01max_factorized_amp,sdxl_t01max_output_moment --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_t01_stylemax
endlocal
