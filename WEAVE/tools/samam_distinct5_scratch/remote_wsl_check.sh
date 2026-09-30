#!/usr/bin/env bash
echo "=== which python ==="
which python
python --version
echo "=== venvs ==="
ls /root/venvs/ 2>/dev/null || echo "no /root/venvs"
echo "=== conda envs ==="
conda env list 2>/dev/null || echo "no conda"
echo "=== mamba-ssm check ==="
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
echo "=== torch check ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
echo "=== pytorch_lightning ==="
python -c "import pytorch_lightning as pl; print('pl:', pl.__version__)" 2>&1
echo "=== open_clip ==="
python -c "import open_clip; print('open_clip: ok')" 2>&1
echo "=== lpips ==="
python -c "import lpips; print('lpips: ok')" 2>&1
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo "=== samam dir ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/ 2>&1
echo "=== data dir ==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/ 2>&1
echo "=== DONE ==="
