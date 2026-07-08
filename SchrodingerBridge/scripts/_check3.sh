#!/bin/bash
echo "=== build dir .o files ==="
ls -la /tmp/mamba_full/mamba-1.2.2/build/temp.linux-x86_64-3.10/csrc/selective_scan/*.o 2>/dev/null | wc -l
ls -la /tmp/mamba_full/mamba-1.2.2/build/temp.linux-x86_64-3.10/csrc/selective_scan/*.o 2>/dev/null
echo ""
echo "=== build dir .so files ==="
find /tmp/mamba_full/mamba-1.2.2/build -name "*.so" 2>/dev/null
echo ""
echo "=== ninja log ==="
tail -10 /tmp/mamba_full/mamba-1.2.2/build/temp.linux-x86_64-3.10/csrc/selective_scan/build.ninja 2>/dev/null || echo "no ninja file"
echo ""
echo "=== mamba_full.log tail ==="
tail -30 /tmp/mamba_full.log 2>/dev/null
echo ""
echo "=== mamba_build.log tail ==="
tail -30 /tmp/mamba_build.log 2>/dev/null
echo ""
echo "=== ps ninja count ==="
ps -ef | grep -E "nvcc|cc1plus|cicc|ptxas" | grep -v grep | wc -l
echo ""
echo "=== import mamba_ssm ==="
source /root/samam_venv/bin/activate
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | head -3
echo ""
echo "=== check site-packages ==="
ls /root/samam_venv/lib/python3.10/site-packages/ | grep -iE "mamba|selective" 2>/dev/null
