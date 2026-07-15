#!/usr/bin/env bash
source /home/xy/venvs/samam312/bin/activate
pip install lpips 2>&1 | tail -10
echo "=== lpips check ==="
python -c "import lpips; print('lpips OK', lpips.__version__)"
