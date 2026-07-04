@echo off
echo Launching SaMam in WSL mamba-ssm environment...
wsl -d Ubuntu-22.04 -- /home/xy/venvs/samam312/bin/python /mnt/i/GitHub/Latent_Style/SchrodingerBridge/tools/remote_samam_wsl.py
echo ==SAMAM_WSL_DONE==
