#!/bin/bash
echo "=== Python version ==="
which python3
python3 --version

echo "=== Conda ==="
which conda 2>/dev/null && conda env list 2>/dev/null || echo "NO_CONDA"

echo "=== Torch ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1

echo "=== SchrodingerBridge import ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python3 -c "import src; print('OK:', src)" 2>&1

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader 2>&1

echo "=== Run entry test ==="
python3 -c "from src.run import main; print('run.main importable')" 2>&1
