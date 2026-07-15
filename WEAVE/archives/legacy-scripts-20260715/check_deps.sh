#!/bin/bash
echo "=== Check diffusers ==="
/home/xy/venvs/samam312/bin/python -c "import diffusers; print('diffusers:', diffusers.__version__)" 2>&1
echo "=== Check pyiqa ==="
/home/xy/venvs/samam312/bin/python -c "import pyiqa; print('pyiqa:', pyiqa.__version__)" 2>&1
echo "=== Check transformers ==="
/home/xy/venvs/samam312/bin/python -c "import transformers; print('transformers:', transformers.__version__)" 2>&1
echo "=== Check mamba_ssm ==="
/home/xy/venvs/samam312/bin/python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
echo "=== Check torch ==="
/home/xy/venvs/samam312/bin/python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
echo "=== Check existing samst-latent checkpoint ==="
ls -la /mnt/i/exp_samst_latent/ 2>&1
echo "=== Check pip list relevant ==="
/home/xy/venvs/samam312/bin/pip list 2>&1 | grep -iE "diffusers|pyiqa|transformers|accelerate"
