#!/bin/bash
# Find Python environments in WSL
echo "=== which python3 ==="
which python3 2>/dev/null
echo "=== python3 version ==="
python3 --version 2>/dev/null

echo "=== conda locations ==="
find / -name "conda" -type f 2>/dev/null | head -5
ls /opt/conda 2>/dev/null
ls ~/miniconda3 2>/dev/null
ls ~/anaconda3 2>/dev/null

echo "=== pip packages ==="
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>&1

echo "=== venvs ==="
find /home -maxdepth 3 -name "activate" 2>/dev/null | head -5
find /mnt/i -maxdepth 4 -name "activate" 2>/dev/null | head -5

echo "=== ls /home ==="
ls /home/ 2>/dev/null

echo "=== whoami ==="
whoami