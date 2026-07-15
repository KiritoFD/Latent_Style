#!/usr/bin/env bash
set -e
source /home/xy/venvs/samam312/bin/activate
echo "=== python ==="
which python
python --version
echo "=== torch ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "=== mamba_ssm ==="
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
echo "=== pytorch_lightning ==="
python -c "import pytorch_lightning as pl; print('pl:', pl.__version__)"
echo "=== open_clip ==="
python -c "import open_clip; print('open_clip: ok')"
echo "=== lpips ==="
python -c "import lpips; print('lpips: ok')"
echo "=== lightning_fabric ==="
python -c "from lightning_fabric import LightningFabric; print('lightning_fabric: ok')"
echo "=== CUDA_HOME ==="
echo $CUDA_HOME
echo "=== nvcc ==="
which nvcc 2>/dev/null && nvcc --version 2>/dev/null | tail -2 || echo "no nvcc"
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo "=== data dir ==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/ 2>&1
echo "=== data train ==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/train/ 2>&1
echo "=== sample train count ==="
for s in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    c=$(ls /mnt/i/wikiart_distinct5_samam_512_classview/train/$s 2>/dev/null | wc -l)
    echo "$s: $c"
done
echo "=== DONE ==="
