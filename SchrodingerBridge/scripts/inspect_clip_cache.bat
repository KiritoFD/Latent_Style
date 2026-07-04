@echo off
echo === CLIP snapshot files ===
dir /b "I:\Github\Latent_Style\eval_cache\hf\models--openai--clip-vit-base-patch32\snapshots\c237dc49a33fc61debc9276459120b7eac67e7ef" 2>nul
echo === Search for preprocessor_config.json ===
dir /s /b "I:\Github\Latent_Style\eval_cache\hf\models--openai--clip-vit-base-patch32\*preprocessor_config.json" 2>nul
echo === Refs ===
dir /b "I:\Github\Latent_Style\eval_cache\hf\models--openai--clip-vit-base-patch32\refs" 2>nul
echo === Local manual_clip ===
dir /b "G:\Github\Latent_Style\eval_cache\manual_clip\openai-clip-vit-base-patch32" 2>nul
echo === Local manual_clip (try I:) ===
dir /b "I:\Github\Latent_Style\eval_cache\manual_clip\openai-clip-vit-base-patch32" 2>nul
echo === DONE ===
