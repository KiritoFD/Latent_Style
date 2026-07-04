@echo off
echo === Launching SaMam 256 WSL job on REMOTE (background) ===
wsl -- bash -lc "nohup bash /mnt/i/Github/Latent_Style/SchrodingerBridge/run_samam_256_wsl.sh > /tmp/samam_256_nohup.log 2>&1 &"
echo LAUNCHED_EXIT_CODE=%ERRORLEVEL%
echo === Wait 5s and check ===
timeout /t 5 /nobreak >nul
wsl -- bash -lc "ls -la /tmp/samam_256_nohup.log 2>&1 && tail -5 /tmp/samam_256_nohup.log 2>&1"
echo === Check log file ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log 2>&1 && tail -10 /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log 2>&1"
echo === DONE ===
