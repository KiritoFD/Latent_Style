@echo off
REM Probe remote WSL environment for latent migration prerequisites
setlocal
set HOST=administrator@100.115.18.62
set PORT=2222

echo === Checking remote WSL environment ===
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'echo === samam312 venv ===; /home/xy/venvs/samam312/bin/python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\" 2>&1 | head -5'"

echo.
echo === Checking latent data ===
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'ls -la /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_latent256/train/ 2>&1 | head -10; echo ---; ls /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/ 2>&1 | head -10'"

echo.
echo === Checking SaMam repo ===
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/train_SaMam_latent.py 2>&1; ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/lightning_module/latent_lightningmodel.py 2>&1'"

echo.
echo === Checking test set ===
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'ls /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_classview/test/ 2>&1 | head -10'"

echo.
echo === Checking VAE cache ===
ssh -p %PORT% -o LogLevel=ERROR %HOST% "wsl -- bash -lc 'ls /mnt/i/Github/Latent_Style/eval_cache/hf/ 2>&1 | head -5'"

endlocal
