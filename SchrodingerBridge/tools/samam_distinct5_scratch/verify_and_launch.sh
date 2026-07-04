#!/usr/bin/env bash
echo "=== VERIFY SYSTEMD DISABLED ==="
echo "PID1=$(ps -p 1 -o comm= 2>/dev/null)"
echo "USER=$(whoami)"
echo "HOME=$HOME"
echo "NO_SYSTEMD_ERROR=$([ ! -e /run/systemd/system ] && echo YES || echo NO)"
echo ""
echo "=== VENV CHECK ==="
source /home/xy/venvs/samam312/bin/activate
python -c "import torch; print('torch=', torch.__version__); import mamba_ssm; print('mamba=', mamba_ssm.__version__)" 2>&1
echo ""
echo "=== GPU CHECK ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "=== READY FOR TRAINING ==="
