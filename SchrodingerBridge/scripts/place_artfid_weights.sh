#!/bin/bash
mkdir -p /tmp/artfid
cp /mnt/c/Users/Administrator/art_inception.pth /tmp/artfid/
ls -la /tmp/artfid/
echo "=== Verify ART-FID loads ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
/home/xy/venvs/samam312/bin/python -c "
import sys
sys.path.insert(0, 'src')
from utils.artfid_metric import load_artfid_feature_extractor
import torch
feat = load_artfid_feature_extractor(device='cpu')
print('ART-FID feature extractor loaded OK')
"
