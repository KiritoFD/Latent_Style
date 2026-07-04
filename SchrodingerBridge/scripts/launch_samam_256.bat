@echo off
echo === Uploading SaMam 256 scripts to remote I drive ===
echo (Scripts will be at I:\Github\Latent_Style\SchrodingerBridge\)
echo === Launching SaMam 256 WSL job in background ===
wsl -- bash -lc "nohup bash /mnt/i/Github/Latent_Style/SchrodingerBridge/run_samam_256_wsl.sh > /tmp/samam_256_nohup.log 2>&1 &"
echo LAUNCHED
echo === Check status with: wsl -- bash -lc "tail -30 /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log" ===
