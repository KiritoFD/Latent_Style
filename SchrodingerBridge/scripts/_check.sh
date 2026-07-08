#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== pip list ==="
pip list 2>/dev/null | grep -iE "mamba|causal|torch|triton|ninja"
echo "=== processes ==="
ps -ef | grep -E "nvcc|cc1plus|ninja|pip" | grep -v grep | head -10
echo "=== import mamba_ssm ==="
python -c "import mamba_ssm; print(mamba_ssm.__version__)" 2>&1 | head -5
echo "=== selective_scan_fn ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1 | head -5
echo "=== install log tail ==="
ls -la /tmp/mamba_install*.log 2>/dev/null
tail -20 /tmp/mamba_install.log 2>/dev/null
