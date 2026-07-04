@echo off
echo === nohup log ===
wsl -- bash -lc "cat /tmp/samam_256_nohup.log 2>&1 | tail -30"
echo === script exists ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/run_samam_256_wsl.sh"
echo === gen script exists ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/gen_samam_single_ckpt.py"
echo === eval script exists ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/eval_samam_metrics_phase2.py"
echo === ckpt exists ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/final_model_20k.ckpt"
echo === python ===
wsl -- bash -lc "ls -la /home/xy/venvs/samam312/bin/python"
echo === test samam python ===
wsl -- bash -lc "/home/xy/venvs/samam312/bin/python -c 'import torch; print(torch.__version__)'"
echo === DONE ===
