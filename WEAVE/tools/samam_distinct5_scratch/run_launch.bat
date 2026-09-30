@echo off
echo === LAUNCHING SAMAM TRAINING ===
wsl -d Ubuntu-22.04 -e bash /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_launch_only.sh
echo === LAUNCH EXIT CODE: %ERRORLEVEL% ===
exit /b 0
