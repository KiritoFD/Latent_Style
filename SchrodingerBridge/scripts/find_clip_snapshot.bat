@echo off
echo === Find clip snapshot ===
wsl -- bash -lc "find /mnt/i/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots -maxdepth 2 -type d 2>&1"
echo === List snapshot files ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots/*/ 2>&1"
echo === DONE ===
