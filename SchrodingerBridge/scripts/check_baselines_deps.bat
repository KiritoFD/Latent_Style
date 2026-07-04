@echo off
echo === Check AdaIN weights ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN/models/vgg_normalised.pth 2>&1"
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN/models/decoder.pth 2>&1"
echo === Check SaMST repo ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/networks/transfer_net.py 2>&1"
echo === Check SaMST ckpt ===
wsl -- bash -lc "ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/checkpoint/repro_5style_train2/ 2>&1"
echo === Check LPIPS in samam312 ===
wsl -- bash -lc "/home/xy/venvs/samam312/bin/python -c 'import lpips; print(\"lpips OK\")' 2>&1"
echo === Check transformers in samam312 ===
wsl -- bash -lc "/home/xy/venvs/samam312/bin/python -c 'from transformers import CLIPModel; print(\"transformers OK\")' 2>&1"
echo === DONE ===
