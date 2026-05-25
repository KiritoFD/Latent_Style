@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_min10g_loss_kin025_swd2,sdxl_min10g_loss_kin05_swd4,sdxl_min10g_loss_content005_swd4,sdxl_min10g_loss_spectral_swd2,sdxl_min10g_loss_micro_macro,sdxl_min10g_arch_res1_swd2,sdxl_min10g_arch_spatial005_swd2,sdxl_min10g_arch_diffeo002_swd2 --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_minimal_loss_arch
endlocal
