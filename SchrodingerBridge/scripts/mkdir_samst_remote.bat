@echo off
echo === Create remote dirs ===
wsl -- bash -lc "mkdir -p /mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/networks"
wsl -- bash -lc "mkdir -p /mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/checkpoint/repro_5style_train2"
echo === DONE ===
