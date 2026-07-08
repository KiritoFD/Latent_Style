#!/bin/bash
# Rebuild mamba_ssm with cusparse.h available
echo "=== Rebuild mamba_ssm ==="
source /root/samam_venv/bin/activate
export MAMBA_FORCE_BUILD=TRUE
export FORCE_BUILD=TRUE
export MAX_JOBS=4
cd /tmp/mamba_full/mamba-1.2.2

# Clean previous build
rm -rf build/

echo "START=$(date)" | tee /tmp/mamba_rebuild2.log
pip install . --no-build-isolation --no-deps 2>&1 | tee -a /tmp/mamba_rebuild2.log
RC=$?
echo "END=$(date) RC=$RC" | tee -a /tmp/mamba_rebuild2.log

echo ""
echo "=== Verify mamba_ssm ==="
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1

echo ""
echo "=== pip list ==="
pip list 2>/dev/null | grep -iE "mamba|causal|torch"

exit $RC
