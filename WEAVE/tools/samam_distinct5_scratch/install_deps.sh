#!/usr/bin/env bash
source /home/xy/venvs/samam312/bin/activate
echo "=== Installing open_clip_torch ==="
pip install open_clip_torch 2>&1 | tail -5
echo ""
echo "=== Checking all deps ==="
python -c "import lpips; print('lpips OK')"
python -c "import open_clip; print('open_clip OK')"
python -c "import torch; print('torch', torch.__version__)"
python -c "import mamba_ssm; print('mamba_ssm OK')"
python -c "import pytorch_lightning as pl; print('pl', pl.__version__)"
echo "=== DONE ==="
