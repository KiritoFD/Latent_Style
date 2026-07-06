#!/usr/bin/env bash
echo "===pixel256 generated images so far==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003 -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l
echo "===sample image info==="
FIRST_IMG=$(find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003 -name "*.png" 2>/dev/null | head -1)
if [ -n "$FIRST_IMG" ]; then
    file "$FIRST_IMG"
    echo "path: $FIRST_IMG"
    # Check if image is all black/white using python
    /home/xy/venvs/samam312/bin/python -c "
from PIL import Image
import numpy as np
img = np.array(Image.open('$FIRST_IMG').convert('RGB'))
print(f'shape: {img.shape}')
print(f'min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}, std: {img.std():.2f}')
print(f'unique values: {len(np.unique(img))}')
if img.std() < 1.0:
    print('WARNING: image is nearly uniform (all black/white)')
elif img.max() < 10:
    print('WARNING: image is very dark')
else:
    print('OK: image has reasonable variation')
" 2>&1
else
    echo "No images generated yet"
fi
