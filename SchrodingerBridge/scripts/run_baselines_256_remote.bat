@echo off
echo === Running Baselines 256 WSL pipeline (foreground) ===
wsl -- bash -lc "bash /mnt/i/Github/Latent_Style/SchrodingerBridge/run_baselines_256_wsl.sh"
echo EXIT_CODE=%ERRORLEVEL%
