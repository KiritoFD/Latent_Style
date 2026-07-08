#!/bin/bash
# Build mamba-ssm 1.2.2 from source (no prebuilt wheel download)
set -e
source /root/samam_venv/bin/activate

echo "=== STEP 1: Download source tarball from PyPI ==="
cd /tmp
pip download mamba-ssm==1.2.2 --no-deps --no-binary=:all: -d /tmp/mamba_src 2>&1 | tail -5
ls -la /tmp/mamba_src/

echo ""
echo "=== STEP 2: Extract source ==="
cd /tmp/mamba_src
TARBALL=$(ls mamba_ssm-1.2.2.tar.gz 2>/dev/null || ls *.tar.gz 2>/dev/null | head -1)
echo "tarball: $TARBALL"
if [ -z "$TARBALL" ]; then
    echo "ERROR: no tarball found"
    ls -la
    exit 1
fi
tar xzf "$TARBALL"
DIR_NAME=$(basename "$TARBALL" .tar.gz)
echo "extracted dir: $DIR_NAME"
cd "$DIR_NAME"
ls -la

echo ""
echo "=== STEP 3: Patch setup.py to skip wheel URL download ==="
# Some versions of mamba setup.py try to download a prebuilt wheel; disable that
if grep -q "Guessing wheel URL" setup.py 2>/dev/null; then
    echo "Found wheel URL guess in setup.py, patching..."
    cp setup.py setup.py.bak
    # Replace the URL guess with a no-op
    python -c "
import re
with open('setup.py', 'r') as f:
    content = f.read()
# Remove the wheel URL guess logic
content = re.sub(r'if not IS_64BIT.*?pass', 'pass', content, flags=re.DOTALL)
# Or just override FORCE_BUILD
content = 'import os\nos.environ[\"MAMBA_FORCE_BUILD\"]=\"1\"\n' + content
with open('setup.py', 'w') as f:
    f.write(content)
print('patched')
"
fi

# Force build via env var
export MAMBA_FORCE_BUILD=1
export FORCE_BUILD=1

echo ""
echo "=== STEP 4: Build & install from source ==="
pip install . --no-build-isolation --no-deps -v 2>&1 | tail -30

echo ""
echo "=== STEP 5: Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== DONE ==="
