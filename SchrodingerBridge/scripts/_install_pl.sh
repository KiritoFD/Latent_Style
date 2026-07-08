#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Install pytorch_lightning and deps ==="
pip install --no-deps pytorch_lightning lightning-utilities torchmetrics packaging PyYAML fsspec tqdm 2>&1 | tail -15

echo ""
echo "=== Test SaMam imports ==="
cd /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam
python -c "
import sys
sys.path.insert(0, '/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam')
from TRAIN.lightning_module.lightningmodel import LightningModel
print('LightningModel OK')
" 2>&1 | tail -10

echo ""
echo "=== pip list ==="
pip list 2>/dev/null | grep -iE "mamba|causal|torch|transformers|lightning|pytorch"
