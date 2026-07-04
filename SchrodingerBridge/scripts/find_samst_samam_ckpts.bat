@echo off
echo === Search SaMam ckpt ===
wsl -- bash -lc "find /mnt/i/Github -name 'final_model.ckpt' -type f 2>/dev/null | head -5"
wsl -- bash -lc "find /mnt/i/Github -name '*.ckpt' -path '*samam*' 2>/dev/null | head -10"
wsl -- bash -lc "find /mnt/i -name 'latest_ckpt.pth' -type f 2>/dev/null | head -5"
wsl -- bash -lc "find /mnt/i -name '*.pth' -path '*samst*' 2>/dev/null | head -10"
wsl -- bash -lc "find /mnt/i -name '*.pth' -path '*SaMST*' 2>/dev/null | head -10"
wsl -- bash -lc "find /mnt/i -name '*.ckpt' -path '*distinct5*' 2>/dev/null | head -20"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/ 2>&1 | head -10"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMST-main/ 2>&1 | head -10"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/ 2>&1 | head -10"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/ 2>&1"
echo === DONE ===
