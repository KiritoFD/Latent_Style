#!/bin/bash
echo "=== Find local CLIP cache ==="
find /mnt/i/Github/Latent_Style -type d -name "*clip-vit-base*" 2>/dev/null | head -5
find /mnt/c/Users/Administrator -type d -name "*clip-vit-base*" 2>/dev/null | head -5
find /home/xy -type d -name "*clip-vit-base*" 2>/dev/null | head -5
echo ""
echo "=== HF cache ==="
ls -la /home/xy/.cache/huggingface/hub/ 2>/dev/null | head -20
echo ""
echo "=== Find pyiqa musiq cache ==="
find /home/xy -name "*.pth" -path "*musiq*" 2>/dev/null | head -5
find /mnt/i -name "*.pth" -path "*musiq*" 2>/dev/null | head -5
