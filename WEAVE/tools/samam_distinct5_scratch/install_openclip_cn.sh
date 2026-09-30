#!/usr/bin/env bash
source /home/xy/venvs/samam312/bin/activate
echo "=== Installing open_clip_torch via tsinghua mirror ==="
pip install --timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple open_clip_torch 2>&1 | tail -10
echo ""
echo "=== Checking open_clip ==="
python -c "import open_clip; print('open_clip OK')"
echo "=== DONE ==="
