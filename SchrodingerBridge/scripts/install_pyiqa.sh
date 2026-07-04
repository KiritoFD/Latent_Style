#!/bin/bash
/home/xy/venvs/samam312/bin/pip install pyiqa 2>&1 | tail -10
echo "=== verify ==="
/home/xy/venvs/samam312/bin/python -c "import pyiqa; print('pyiqa', pyiqa.__version__)" 2>&1
