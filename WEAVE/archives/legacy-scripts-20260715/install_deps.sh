#!/bin/bash
# Install with longer timeout and retry
/home/xy/venvs/samam312/bin/pip install --timeout 300 --retries 5 diffusers transformers accelerate pyiqa 2>&1 | tail -20
echo "=== verify ==="
/home/xy/venvs/samam312/bin/python -c "import diffusers; print('diffusers', diffusers.__version__)" 2>&1
/home/xy/venvs/samam312/bin/python -c "import pyiqa; print('pyiqa OK')" 2>&1
/home/xy/venvs/samam312/bin/python -c "import transformers; print('transformers', transformers.__version__)" 2>&1
