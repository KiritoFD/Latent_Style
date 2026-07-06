#!/bin/bash
# Check torch + GPU on remote
echo "=== python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' ==="
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -15
echo ""
echo "=== pip3 list | grep -i torch ==="
pip3 list 2>/dev/null | grep -i torch
echo ""
echo "=== python3 -m site ==="
python3 -m site 2>&1 | head -10
