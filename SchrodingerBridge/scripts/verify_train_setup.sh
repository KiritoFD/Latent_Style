#!/bin/bash
echo "===TRAIN SCRIPT==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/train_latent256_photo2art.sh
echo ""
echo "===CONFIG KEY FIELDS==="
PYTHON=/home/xy/venvos/samam312/bin/python
$PYTHON -c "
import json
with open('/mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_latent_256_photo2art.json') as f:
    c = json.load(f)
print('data_root:', c['data']['data_root'])
print('latent_cache_dir:', c['data']['latent_cache_dir'])
print('style_subdirs:', c['data']['style_subdirs'])
print('test_image_dir:', c['training']['test_image_dir'])
print('save_dir:', c['checkpoint']['save_dir'])
print('batch_size:', c['training']['batch_size'])
print('num_epochs:', c['training']['num_epochs'])
"
echo "===PACKED CACHE==="
ls -lh /mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/
echo "===TEST DIR==="
ls /mnt/i/legacy256_overfit50/test/ | head -10
