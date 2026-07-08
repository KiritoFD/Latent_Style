#!/bin/bash
# Install mamba-ssm from local source tarball (no GitHub access needed)
set -e
source /root/samam_venv/bin/activate

TARBALL="/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/mamba_ssm-1.2.2.tar.gz"

echo "=== STEP 1: Verify tarball exists ==="
ls -la "$TARBALL"

echo ""
echo "=== STEP 2: Extract source ==="
WORK_DIR=/tmp/mamba_build
rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
tar xzf "$TARBALL"
ls -la
cd mamba_ssm-1.2.2
ls -la

echo ""
echo "=== STEP 3: Look at setup.py for wheel URL logic ==="
grep -n "Guessing wheel\|FORCE_BUILD\|wheel_url" setup.py 2>&1 | head -10

echo ""
echo "=== STEP 4: Build & install (force build, no wheel URL) ==="
export MAMBA_FORCE_BUILD=1
export FORCE_BUILD=1
# Patch setup.py to skip the wheel URL guess
python -c "
with open('setup.py', 'r') as f:
    content = f.read()
# Find and replace the wheel URL guess block
import re
# Match the wheel URL download logic and replace with pass
# The pattern is typically:
#   if not FORCE_BUILD: try: ... download wheel ... except: pass
# Or it tries to fetch a prebuilt wheel from GitHub
print('Setup.py length:', len(content))
print('Has FORCE_BUILD:', 'FORCE_BUILD' in content)
print('Has wheel URL:', 'wheel_url' in content or 'Guessing wheel' in content)
"
# Just try with FORCE_BUILD env var
pip install . --no-build-isolation --no-deps -v 2>&1 | tail -40

echo ""
echo "=== STEP 5: Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1
python -c "from PIL import Image; print('PIL OK')" 2>&1
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1
python -c "import tqdm; print('tqdm OK')" 2>&1

echo ""
echo "=== DONE ==="
