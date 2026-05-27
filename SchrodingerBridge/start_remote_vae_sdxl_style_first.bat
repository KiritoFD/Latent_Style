@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_stylefirst_swd8_p1357_k025,sdxl_stylefirst_swd12_p1357_k025,sdxl_stylefirst_swd16_p1357_k010,sdxl_stylefirst_spectral12,sdxl_stylefirst_micro_macro,sdxl_stylefirst_spatial010,sdxl_stylefirst_res1,sdxl_stylefirst_diffeo005 --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_style_first
endlocal
