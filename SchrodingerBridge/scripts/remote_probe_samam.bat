@echo off
echo === Remote WSL availability ===
wsl --status 2>&1
echo === WSL distros ===
wsl --list --verbose 2>&1
echo === Check samam312 venv ===
wsl -- bash -lc "ls /home/xy/venvs/samam312/bin/python 2>&1"
echo === Check I drive SchrodingerBridge ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/SchrodingerBridge/ 2>&1"
echo === Check gen_samam_single_ckpt.py ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/gen_samam_single_ckpt.py 2>&1"
echo === Check run_samam_256_wsl.sh ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/run_samam_256_wsl.sh 2>&1"
echo === Check eval_samam_metrics_phase2.py ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/eval_samam_metrics_phase2.py 2>&1"
echo === Check SaMam ckpt ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/final_model_20k.ckpt 2>&1"
echo === Check SaMam repo ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/lightning_module/lightningmodel.py 2>&1"
echo === Check test set ===
wsl -- bash -lc "ls /mnt/i/wikiart_distinct5_samam_512_classview/test/ 2>&1"
echo === Check clip cache ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/eval_cache/hf/ 2>&1"
echo === Check existing nohup log ===
wsl -- bash -lc "ls -la /tmp/samam_256_nohup.log 2>&1"
echo === Check existing eval output ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/exp_samam/eval_256/ 2>&1"
echo === DONE ===
