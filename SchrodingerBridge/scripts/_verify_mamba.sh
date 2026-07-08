#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Test selective_scan_fn direct import ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK:', selective_scan_fn)" 2>&1 | head -10

echo ""
echo "=== If failed, install transformers ==="
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Direct import failed, installing transformers..."
    pip install --no-deps transformers tokenizers huggingface_hub regex safetensors 2>&1 | tail -10
    echo ""
    echo "=== Retest after transformers install ==="
    python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK:', selective_scan_fn)" 2>&1 | head -10
    python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | head -10
fi

echo ""
echo "=== Test causal_conv1d ==="
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>&1

echo ""
echo "=== Test cuda ==="
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1

echo ""
echo "=== Final pip list ==="
pip list 2>/dev/null | grep -iE "mamba|causal|torch|transformers|tokenizers"
