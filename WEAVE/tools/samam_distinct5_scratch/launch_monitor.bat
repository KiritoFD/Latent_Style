@echo off
echo === STARTING MONITOR ===
wsl -d Ubuntu-22.04 -e bash /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/start_monitor.sh
echo === MONITOR LAUNCH DONE ===
exit /b 0
