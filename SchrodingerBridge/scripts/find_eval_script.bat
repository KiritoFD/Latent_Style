@echo off
echo === Search eval_samam_metrics_phase2.py ===
wsl -- bash -lc "find /mnt/i/Github/Latent_Style -name 'eval_samam_metrics_phase2.py' -type f 2>/dev/null"
echo === Search gen_samam_images_phase1.py ===
wsl -- bash -lc "find /mnt/i/Github/Latent_Style -name 'gen_samam_images_phase1.py' -type f 2>/dev/null"
echo === List SchrodingerBridge root ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/ 2>&1"
echo === List tools if exists ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ 2>&1"
echo === List tools/samam_distinct5_scratch if exists ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/ 2>&1"
echo === DONE ===
