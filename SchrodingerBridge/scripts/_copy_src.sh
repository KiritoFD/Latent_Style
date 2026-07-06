#!/usr/bin/env bash
set -uo pipefail
cp /mnt/c/Users/Administrator/blocks620.py /mnt/i/Github/Latent_Style/SchrodingerBridge/src/
cp /mnt/c/Users/Administrator/model620.py /mnt/i/Github/Latent_Style/SchrodingerBridge/src/
echo COPIED
grep -n "attn_mode" /mnt/i/Github/Latent_Style/SchrodingerBridge/src/blocks620.py | head -5
grep -n "attn_mode=self.style_attn_mode" /mnt/i/Github/Latent_Style/SchrodingerBridge/src/model620.py
echo "===GPU BEFORE==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
