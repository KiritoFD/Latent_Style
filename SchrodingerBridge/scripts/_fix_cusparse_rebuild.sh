#!/bin/bash
# Fix missing cusparse.h and rebuild mamba_ssm
# NOTE: no set -e, we want to continue even if some checks fail

echo "=== STEP 1: Check what's in /usr/local/cuda/include ==="
ls /usr/local/cuda/include/cusparse.h 2>&1 || true
ls /usr/local/cuda/include/cublas.h 2>&1 || true
ls /usr/local/cuda/include/curand.h 2>&1 || true

echo ""
echo "=== STEP 2: Check available cuda packages ==="
dpkg -l | grep -iE "cuda|cusparse|cublas" | head -20 || true

echo ""
echo "=== STEP 3: Install missing cusparse dev ==="
apt-get update -qq 2>&1 | tail -3 || true
apt-get install -y libcusparse-dev libcublas-dev libcurand-dev 2>&1 | tail -20

echo ""
echo "=== STEP 4: Verify cusparse.h ==="
find /usr -name "cusparse.h" 2>/dev/null | head -5
ls -la /usr/local/cuda/include/cusparse.h 2>&1 || true

echo ""
echo "=== STEP 5: Re-run pip install ==="
source /root/samam_venv/bin/activate
export MAMBA_FORCE_BUILD=TRUE
export FORCE_BUILD=TRUE
export MAX_JOBS=4
cd /tmp/mamba_full/mamba-1.2.2

# Clean previous build
rm -rf build/

echo "START=$(date)" | tee /tmp/mamba_rebuild.log
pip install . --no-build-isolation --no-deps 2>&1 | tee -a /tmp/mamba_rebuild.log
RC=$?
echo "END=$(date) RC=$RC" | tee -a /tmp/mamba_rebuild.log

echo ""
echo "=== STEP 6: Verify mamba_ssm ==="
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1

exit $RC
