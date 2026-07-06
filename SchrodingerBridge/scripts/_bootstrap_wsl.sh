#!/bin/bash
# Bootstrap WSL environment for mamba_ssm
set -e

echo "=== Update apt ==="
sudo apt-get update -y 2>&1 | tail -3 || apt-get update -y 2>&1 | tail -3

echo ""
echo "=== Install python3-pip ==="
sudo apt-get install -y python3-pip python3-venv 2>&1 | tail -3 || apt-get install -y python3-pip python3-venv 2>&1 | tail -3

echo ""
echo "=== Create venv ==="
python3 -m venv ~/samam_venv
source ~/samam_venv/bin/activate
pip install --upgrade pip 2>&1 | tail -2

echo ""
echo "=== Install torch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo ""
echo "=== Install mamba_ssm deps ==="
pip install packaging 2>&1 | tail -2
pip install causal-conv1d 2>&1 | tail -3
pip install mamba-ssm 2>&1 | tail -5

echo ""
echo "=== Install PIL ==="
pip install Pillow 2>&1 | tail -2

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "from PIL import Image; print('PIL OK')"

echo ""
echo "=== VENV path ==="
echo ~/samam_venv/bin/python

echo "=== DONE ==="
