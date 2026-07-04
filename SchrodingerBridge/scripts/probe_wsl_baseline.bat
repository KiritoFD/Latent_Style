@echo off
echo === WSL Python check ===
wsl -- bash -lc "which python3"
wsl -- bash -lc "python3 --version"
wsl -- bash -lc "python3 -c 'import torch; print(\"torch=\",torch.__version__)' 2>&1"
wsl -- bash -lc "python3 -c 'import mamba_ssm; print(\"mamba_ssm=\",mamba_ssm.__version__)' 2>&1"
wsl -- bash -lc "python3 -c 'import causal_conv1d; print(\"causal_conv1d=\",causal_conv1d.__version__)' 2>&1"
wsl -- bash -lc "ls /home/xy/venvs/samam312/bin/python 2>&1"
wsl -- bash -lc "/home/xy/venvs/samam312/bin/python -c 'import mamba_ssm; print(\"samam312 mamba=\",mamba_ssm.__version__)' 2>&1"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/train_SaMam.py 2>&1"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMST-main/test_model/test/test.py 2>&1"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/checkpoint/repro_5style_train2/ 2>&1 | head -20"
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/ 2>&1 | tail -10"
echo === DONE ===
