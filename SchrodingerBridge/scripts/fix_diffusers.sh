#!/bin/bash
# Downgrade diffusers to compatible version for torch 2.4.0
/home/xy/venvs/samam312/bin/pip install --timeout 300 --retries 5 "diffusers==0.27.2" 2>&1 | tail -20
echo "===verify==="
/home/xy/venvs/samam312/bin/python -c "
from diffusers import AutoencoderKL
import diffusers
print('diffusers:', diffusers.__version__)
print('AutoencoderKL imported OK')
"
