@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
py tools\experiments\run_vae_backend_256_probe.py --variants sdxl_min10g_swd2_p13,sdxl_min10g_swd4_p13,sdxl_min10g_swd2_p135,sdxl_min10g_swd2_p13_decode010,sdxl_min10g_swd2_p13_decode016,sdxl_min10g_swd2_p13_model010,sdxl_min10g_swd2_p13_model016,sdxl_min10g_swd2_p13_steps16 --epochs 8 --eval-epochs 6,7,8 --skip-existing-latents --out-root exp\vae_backend_256_sdxl_minimal_scale_switches
endlocal
