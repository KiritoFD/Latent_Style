#!/bin/bash
# Find mamba_ssm in WSL
echo "=== conda envs ==="
which conda 2>&1
conda env list 2>&1

echo ""
echo "=== Find mamba_ssm anywhere ==="
find / -name "mamba_ssm" -type d 2>/dev/null | head -10

echo ""
echo "=== Find conda python ==="
ls /opt/conda/bin/python* 2>&1
ls /home/*/miniconda3/bin/python* 2>&1
ls /home/*/anaconda3/bin/python* 2>&1

echo ""
echo "=== Find any python with mamba_ssm ==="
for p in /opt/conda/envs/*/bin/python /home/*/miniconda3/envs/*/bin/python /home/*/anaconda3/envs/*/bin/python /root/miniconda3/envs/*/bin/python; do
    if [ -x "$p" ]; then
        echo "Trying $p"
        $p -c "import mamba_ssm; print('  mamba_ssm OK:', mamba_ssm.__version__)" 2>&1 | head -2
    fi
done

echo ""
echo "=== pip list of system python3 ==="
pip3 list 2>/dev/null | grep -iE "mamba|torch|selective" | head -10

echo ""
echo "=== Home dirs ==="
ls /home/ 2>&1
ls /root/ 2>&1 | head -5
