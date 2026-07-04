@echo off
echo === Running SaMam 256 WSL job (foreground, SSH will hold) ===
wsl -- bash -lc "bash /mnt/i/Github/Latent_Style/SchrodingerBridge/run_samam_256_wsl.sh"
echo EXIT_CODE=%ERRORLEVEL%
