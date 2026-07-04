@echo off
echo === LATENT256 TRAIN DIR ===
dir /b "I:\wikiart_distinct5_samam_512_latent256\train" 2>nul
echo === STYLE SUBDIR CONTENT (Early_Renaissance) ===
dir /b "I:\wikiart_distinct5_samam_512_latent256\train\Early_Renaissance" 2>nul | find /c /v ""
echo === SAMPLE FILE NAMES ===
dir /b "I:\wikiart_distinct5_samam_512_latent256\train\Early_Renaissance" 2>nul | findstr /n "^" | findstr "^[1-3]:"
echo === PACKED CACHE DIR ===
dir /b "I:\wikiart_distinct5_samam_512_latent256\train\.latent_cache\packed" 2>nul | findstr /n "^" | findstr "^[1-5]:"
echo === TEST DIR (classview) ===
dir /b "I:\wikiart_distinct5_samam_512_classview\test" 2>nul
echo === DONE ===
