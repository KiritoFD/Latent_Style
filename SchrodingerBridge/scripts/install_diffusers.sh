#!/bin/bash
/home/xy/venvs/samam312/bin/pip install diffusers transformers accelerate 2>&1 | tail -10
echo "=== verify ==="
/home/xy/venvs/samam312/bin/python -c "import diffusers; print('diffusers', diffusers.__version__)" 2>&1
/home/xy/venvs/samam312/bin/python -c "import pyiqa; print('pyiqa OK')" 2>&1
