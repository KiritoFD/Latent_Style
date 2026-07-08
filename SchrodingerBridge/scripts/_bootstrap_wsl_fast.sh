#!/bin/bash
# Fast bootstrap: use --no-build-isolation to reuse installed torch
set -e
source /root/samam_venv/bin/activate

echo "=== current packages ==="
pip list 2>&1 | grep -E "torch|packaging|causal|mamba|pillow|triton"

echo ""
echo "=== install build tools (no isolation) ==="
pip install setuptools wheel ninja 2>&1 | tail -3

echo ""
echo "=== install causal-conv1d (no-build-isolation, pre-built wheel preferred) ==="
# Try prebuilt wheel first
pip install causal-conv1d --no-build-isolation 2>&1 | tail -10

echo ""
echo "=== install mamba-ssm (no-build-isolation) ==="
pip install mamba-ssm --no-build-isolation 2>&1 | tail -10

echo ""
echo "=== install tqdm ==="
pip install tqdm 2>&1 | tail -2

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import tqdm; print('tqdm OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"

echo ""
echo "=== DONE ==="
