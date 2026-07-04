#!/bin/bash
echo "=== Full log with stderr ==="
cat /mnt/i/exp_extra_metrics.log 2>/dev/null
echo ""
echo "=== Try ART-FID manually on samst_latent ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
/home/xy/venvs/samam312/bin/python -c "
import sys
sys.path.insert(0, 'src')
from utils.artfid_metric import load_artfid_feature_extractor
import torch
try:
    feat = load_artfid_feature_extractor(device='cpu')
    print('ART-FID feature extractor loaded OK')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
"
