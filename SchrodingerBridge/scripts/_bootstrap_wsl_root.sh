#!/bin/bash
# Bootstrap WSL env with root - no sudo needed
set -e
echo "=== whoami ==="
whoami
echo "=== python version ==="
python3 --version
which python3
which pip3 || echo "pip3 not found at default path"
pip3 --version 2>&1 | head -2 || echo "pip3 --version failed"

echo ""
echo "=== apt install python3-venv (if needed) ==="
apt-get install -y python3-venv python3-pip 2>&1 | tail -3

echo ""
echo "=== Create venv ==="
python3 -m venv /root/samam_venv
source /root/samam_venv/bin/activate
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
echo "=== Install Pillow tqdm ==="
pip install Pillow tqdm 2>&1 | tail -2

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "from PIL import Image; print('PIL OK')"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== VENV path ==="
echo /root/samam_venv/bin/python
echo "=== DONE ==="
