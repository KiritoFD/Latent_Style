#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Install requests ==="
pip install --no-deps requests urllib3 charset-normalizer idna 2>&1 | tail -10

echo ""
echo "=== Test selective_scan_fn import ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>&1 | tail -5

echo ""
echo "=== Test full mamba_ssm import ==="
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | tail -5

echo ""
echo "=== Final pip list ==="
pip list 2>/dev/null | grep -iE "mamba|causal|torch|transformers|requests|tokenizers"
