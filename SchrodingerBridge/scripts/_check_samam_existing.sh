#!/bin/bash
echo "=== existing samam images ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/ 2>/dev/null | head -10
echo "..."
echo "count:"
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/ 2>/dev/null | wc -l
echo ""
echo "=== sample image dim (first one) ==="
FIRST=$(ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/*.png 2>/dev/null | head -1)
echo "first: $FIRST"
if [ -n "$FIRST" ]; then
    /root/samam_venv/bin/python -c "from PIL import Image; im=Image.open('$FIRST'); print('size:', im.size)" 2>&1
fi
echo ""
echo "=== _DONE marker? ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/_DONE 2>/dev/null || echo "no _DONE"

echo ""
echo "=== torch install log tail (last 30 lines) ==="
# this won't show much because pip output is buffered, but try
ls -la /root/samam_venv/lib/python3.10/site-packages/torch/ 2>/dev/null | head -5 || echo "torch not yet installed"
