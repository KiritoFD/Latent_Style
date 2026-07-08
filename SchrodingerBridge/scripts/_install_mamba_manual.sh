#!/bin/bash
# Manually download mamba-ssm wheel and install
set -e
source /root/samam_venv/bin/activate

WHL_URL="https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
WHL_LOCAL="/tmp/mamba_ssm-1.2.2.whl"

echo "=== Download wheel ==="
echo "URL: $WHL_URL"
# Try multiple methods
curl -L -o "$WHL_LOCAL" "$WHL_URL" 2>&1 | tail -5 || wget -O "$WHL_LOCAL" "$WHL_URL" 2>&1 | tail -5

echo ""
echo "=== Check downloaded file ==="
ls -la "$WHL_LOCAL" 2>&1
file "$WHL_LOCAL" 2>&1

echo ""
echo "=== Install wheel ==="
pip install "$WHL_LOCAL" --no-build-isolation --no-deps 2>&1 | tail -10

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
