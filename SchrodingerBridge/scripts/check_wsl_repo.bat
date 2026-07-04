@echo off
echo === Check WSL SchrodingerBridge repo ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/ 2>&1 | head -20"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ 2>&1 | head -10"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/ 2>&1 | head -10"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/Related_Works/repos/SaMam/TRAIN/lightning_module/ 2>&1"
wsl -- bash -lc "ls /mnt/g/GitHub/Latent_Style/SchrodingerBridge/ 2>&1 | head -5"
echo === DONE ===
