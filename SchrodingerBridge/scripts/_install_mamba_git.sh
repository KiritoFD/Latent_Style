#!/bin/bash
# Clone mamba repo, checkout v1.2.2, force build from source
set -e
source /root/samam_venv/bin/activate

echo "=== STEP 1: Clone mamba repo ==="
cd /tmp
rm -rf mamba_src
git clone --depth 1 --branch v1.2.2 https://github.com/state-spaces/mamba.git mamba_src 2>&1 | tail -5
ls -la /tmp/mamba_src/

echo ""
echo "=== STEP 2: Check setup.py for wheel URL logic ==="
grep -n "Guessing wheel\|FORCE_BUILD\|wheel_url\|pyproject.toml" /tmp/mamba_src/setup.py 2>&1 | head -20

echo ""
echo "=== STEP 3: Force build & install ==="
cd /tmp/mamba_src
export MAMBA_FORCE_BUILD=1
export FORCE_BUILD=1
pip install . --no-build-isolation --no-deps 2>&1 | tail -20

echo ""
echo "=== STEP 4: Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== DONE ==="
