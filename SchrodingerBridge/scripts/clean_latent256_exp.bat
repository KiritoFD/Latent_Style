@echo off
echo === Cleaning latent256 exp dir ===
if exist "C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10" (
  rmdir /s /q "C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10"
  echo CLEANED
) else (
  echo NOT_EXISTS
)
echo === Verifying CLIP cache ===
dir /b "I:\Github\Latent_Style\eval_cache\hf" 2>nul
echo === Verifying CLIP cache hub ===
dir /b "I:\Github\Latent_Style\eval_cache\hf\hub" 2>nul
echo === DONE ===
