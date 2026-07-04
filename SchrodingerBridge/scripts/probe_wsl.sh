#!/bin/bash
echo "=== venv ==="
/home/xy/venvs/samam312/bin/python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
echo "=== latent train dir ==="
ls /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_latent256/train/
echo "=== packed cache ==="
ls /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/ 2>&1
echo "=== test set ==="
ls /mnt/i/Github/Latent_Style/wikiart_distinct5_samam_512_classview/test/
echo "=== VAE cache ==="
ls /mnt/i/Github/Latent_Style/eval_cache/hf/ 2>&1
echo "=== mamba_ssm ==="
/home/xy/venvs/samam312/bin/python -c "import mamba_ssm; print('mamba_ssm OK', mamba_ssm.__version__)" 2>&1
