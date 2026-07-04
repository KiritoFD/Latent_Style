@echo off
echo === Clean old baseline_256 output ===
wsl -- bash -lc "rm -rf /mnt/i/Github/Latent_Style/exp_baseline_256/adain /mnt/i/Github/Latent_Style/exp_baseline_256/wct /mnt/i/Github/Latent_Style/exp_baseline_256/samst /mnt/i/Github/Latent_Style/exp_baseline_256/baseline_256.log 2>&1"
echo === Rerun pipeline ===
wsl -- bash -lc "bash /mnt/i/Github/Latent_Style/SchrodingerBridge/run_baselines_256_wsl.sh"
echo EXIT_CODE=%ERRORLEVEL%
