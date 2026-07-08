#!/bin/bash
# Install mamba-ssm 1.2.2 (compatible with torch 2.5.1)
set -e
source /root/samam_venv/bin/activate

echo "=== Install mamba-ssm 1.2.2 (no-build-isolation) ==="
pip install mamba-ssm==1.2.2 --no-build-isolation 2>&1 | tail -15

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== DONE ==="
