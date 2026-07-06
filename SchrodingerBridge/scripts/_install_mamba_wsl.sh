#!/bin/bash
# Install mamba_ssm in WSL
set -e

echo "=== System info ==="
python3 --version
echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"

echo ""
echo "=== Install pip packages ==="
pip3 install --user --upgrade pip 2>&1 | tail -3

echo ""
echo "=== Check existing torch ==="
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'cuda_ver:', torch.version.cuda)" 2>&1

echo ""
echo "=== Install torch if missing ==="
python3 -c "import torch" 2>&1 || pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo ""
echo "=== Install mamba_ssm deps ==="
pip3 install --user "causal-conv1d>=1.1.0" 2>&1 | tail -3
pip3 install --user mamba-ssm 2>&1 | tail -5

echo ""
echo "=== Verify ==="
python3 -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python3 -c "from PIL import Image; print('PIL OK')" 2>&1
python3 -c "from torchvision.utils import save_image; print('torchvision OK')" 2>&1

echo ""
echo "=== DONE ==="
