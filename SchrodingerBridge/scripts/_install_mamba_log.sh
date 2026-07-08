#!/bin/bash
# Install mamba-ssm with MAMBA_FORCE_BUILD=TRUE and capture full build log
set -e
source /root/samam_venv/bin/activate

TARBALL="/mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/mamba_ssm-1.2.2.tar.gz"
WORK_DIR=/tmp/mamba_build2
LOG=/tmp/mamba_build.log

rm -rf $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR
tar xzf "$TARBALL"
cd mamba_ssm-1.2.2

echo "=== Building with MAMBA_FORCE_BUILD=TRUE ==="
export MAMBA_FORCE_BUILD=TRUE
export FORCE_BUILD=TRUE

# Run pip install with full verbose logging
pip install . --no-build-isolation --no-deps -v > $LOG 2>&1
EXIT=$?
echo "pip exit: $EXIT"

echo ""
echo "=== LAST 60 lines of build log ==="
tail -60 $LOG

echo ""
echo "=== ERROR lines from build log ==="
grep -iE "error:|fatal|undefined|Failed|cannot find" $LOG | head -30

echo ""
echo "=== Verify ==="
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1

echo "=== DONE ==="
