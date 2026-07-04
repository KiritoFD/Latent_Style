#!/bin/bash
/home/xy/venvs/samam312/bin/pip install --timeout 300 --retries 5 "diffusers==0.30.1" 2>&1 | tail -10
echo "===verify==="
/home/xy/venvs/samam312/bin/python -c "
from diffusers import AutoencoderKL
import diffusers
print('diffusers:', diffusers.__version__)
print('AutoencoderKL imported OK')
"
