@echo off
powershell -NoProfile -Command "Get-Content -Path 'C:\Users\Administrator\logs\latent256_train.log' -Tail 30"
echo === GPU ===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo === CKPT LIST ===
dir /b "C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10\*.pt" 2>nul
echo === EVAL DIRS ===
dir /b "C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10\full_eval" 2>nul
echo === DONE ===
