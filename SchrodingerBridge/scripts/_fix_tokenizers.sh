#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Downgrade tokenizers ==="
pip install --no-deps "tokenizers>=0.20,<0.21" 2>&1 | tail -10

echo ""
echo "=== Test selective_scan_fn import ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1 | tail -5

echo ""
echo "=== Test full mamba_ssm import ==="
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | tail -5

echo ""
echo "=== Test selective_scan_fn + cuda ==="
python -c "
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('selective_scan_fn:', selective_scan_fn)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
" 2>&1 | tail -10
