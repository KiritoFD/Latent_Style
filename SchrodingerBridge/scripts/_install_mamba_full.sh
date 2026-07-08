#!/bin/bash
# Install mamba-ssm from full GitHub source (includes csrc/)
set -e
source /root/samam_venv/bin/activate

TARBALL="/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/mamba_v1.2.2_full.tar.gz"
WORK_DIR=/tmp/mamba_full
LOG=/tmp/mamba_full.log

rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
tar xzf "$TARBALL"
ls -la
# Source from GitHub is typically mamba-1.2.2 (not mamba_ssm-1.2.2)
cd mamba-1.2.2
ls -la

echo ""
echo "=== Check csrc dir exists ==="
ls csrc/selective_scan/ | head -10

echo ""
echo "=== Build & install (MAMBA_FORCE_BUILD=TRUE) ==="
export MAMBA_FORCE_BUILD=TRUE
export FORCE_BUILD=TRUE
pip install . --no-build-isolation --no-deps > $LOG 2>&1
EXIT=$?
echo "pip exit: $EXIT"

echo ""
echo "=== LAST 30 lines of build log ==="
tail -30 $LOG

echo ""
echo "=== Error lines ==="
grep -iE "error:|fatal|undefined|cannot|Failed" $LOG | head -20

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1
python -c "from PIL import Image; print('PIL OK')" 2>&1
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1
python -c "import tqdm; print('tqdm OK')" 2>&1

echo ""
echo "=== DONE ==="
