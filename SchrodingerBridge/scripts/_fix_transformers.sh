#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Show full import error ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1

echo ""
echo "=== Downgrade transformers to 4.x ==="
pip install --no-deps "transformers==4.46.3" 2>&1 | tail -10

echo ""
echo "=== Retest ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1 | tail -5
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | tail -5
