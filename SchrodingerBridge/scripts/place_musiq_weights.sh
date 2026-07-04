#!/bin/bash
# Place MUSIQ weights in remote torch hub cache
REMOTE_CACHE=/home/xy/.cache/torch/hub/pyiqa
mkdir -p $REMOTE_CACHE
cp /mnt/c/Users/Administrator/musiq_koniq_ckpt-e95806b9.pth $REMOTE_CACHE/
echo "MUSIQ weights placed at:"
ls -la $REMOTE_CACHE/

# Verify it loads
/home/xy/venvs/samam312/bin/python -c "
import pyiqa
m = pyiqa.create_metric('musiq', device='cpu')
print('MUSIQ loaded OK on remote')
"
