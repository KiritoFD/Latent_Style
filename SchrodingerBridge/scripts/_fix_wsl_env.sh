#!/bin/bash
# Fix: roll back torch to 2.5.1, install mamba-ssm 1.2.2 (compatible)
set -e
source /root/samam_venv/bin/activate

echo "=== STEP 1: Uninstall broken mamba-ssm + transformers + torch 2.12 ==="
pip uninstall -y mamba-ssm transformers tokenizers 2>&1 | tail -5

echo ""
echo "=== STEP 2: Force reinstall torch 2.5.1 + torchvision 0.20.1 ==="
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo ""
echo "=== STEP 3: Verify torch 2.5.1 ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo ""
echo "=== STEP 4: Reinstall causal-conv1d (rebuild for torch 2.5.1 ABI) ==="
pip uninstall -y causal-conv1d 2>&1 | tail -2
pip install causal-conv1d==1.5.0.post8 --no-build-isolation 2>&1 | tail -10

echo ""
echo "=== STEP 5: Install mamba-ssm 1.2.2.post1 (compatible with torch 2.5.1) ==="
pip install mamba-ssm==1.2.2.post1 --no-build-isolation 2>&1 | tail -15

echo ""
echo "=== STEP 6: Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== DONE ==="
