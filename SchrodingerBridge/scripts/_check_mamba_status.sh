#!/bin/bash
echo "=== Check mamba-ssm compile status ==="
echo "--- pip list (mamba/causal/torch) ---"
source /root/samam_venv/bin/activate
pip list 2>/dev/null | grep -iE "mamba|causal|torch|triton|ninja"

echo "--- Check running nvcc/gcc processes ---"
ps -ef | grep -E "nvcc|gcc|cc1plus|ninja|pip" | grep -v grep | head -20

echo "--- Check if mamba_ssm installed ---"
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1 | head -5

echo "--- Check selective_scan_fn ---"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1 | head -5

echo "--- Check compile log if exists ---"
if [ -f /tmp/mamba_install.log ]; then
    echo "Last 20 lines of /tmp/mamba_install.log:"
    tail -20 /tmp/mamba_install.log
fi
