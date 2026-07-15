@echo off
echo === TRAIN SCRIPT ===
wsl -d Ubuntu-22.04 -e bash -c "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh 2>&1; ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/train_SaMam.py 2>&1"
echo === DATA PATHS ===
wsl -d Ubuntu-22.04 -e bash -c "ls /mnt/i/wikiart_distinct5_samam_512_flat/train_flat/content 2>/dev/null | wc -l; ls /mnt/i/wikiart_distinct5_samam_512_flat/test_flat/content 2>/dev/null | wc -l"
echo === VENV ===
wsl -d Ubuntu-22.04 -e bash -c "source /home/xy/venvs/samam312/bin/activate && python -c 'import torch; print(torch.__version__); import mamba_ssm; print(mamba_ssm.__version__)' 2>&1"
echo === OUTPUT DIR ===
wsl -d Ubuntu-22.04 -e bash -c "ls -la /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/ 2>&1"
exit /b 0
