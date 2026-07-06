#!/bin/bash
# Find conda envs in WSL

echo "=== conda envs ==="
conda env list 2>&1 || echo "conda not found"

echo ""
echo "=== /opt/conda ==="
ls /opt/conda/envs/ 2>&1 || echo "no /opt/conda"

echo ""
echo "=== ~/miniconda3 ==="
ls ~/miniconda3/envs/ 2>&1 || echo "no ~/miniconda3"

echo ""
echo "=== ~/anaconda3 ==="
ls ~/anaconda3/envs/ 2>&1 || echo "no ~/anaconda3"

echo ""
echo "=== ~/.conda/envs ==="
ls ~/.conda/envs/ 2>&1 || echo "no ~/.conda/envs"

echo ""
echo "=== find python with mamba_ssm ==="
for py in /opt/conda/envs/*/bin/python ~/miniconda3/envs/*/bin/python ~/anaconda3/envs/*/bin/python ~/.conda/envs/*/bin/python /usr/local/bin/python*; do
    if [ -x "$py" ] 2>/dev/null; then
        ver=$($py --version 2>&1)
        mamba=$($py -c "import mamba_ssm; print(mamba_ssm.__version__)" 2>&1)
        torch=$($py -c "import torch; print(torch.__version__)" 2>&1)
        echo "$py: $ver | mamba=$mamba | torch=$torch"
    fi
done

echo ""
echo "=== check /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam environment.yml ==="
cat /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/environment.yml 2>&1 | head -20
