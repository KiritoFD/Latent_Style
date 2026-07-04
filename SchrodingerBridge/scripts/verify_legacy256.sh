#!/usr/bin/env bash
echo "=== legacy256_overfit50/test/ ==="
for d in /mnt/i/legacy256_overfit50/test/*/; do
    name=$(basename "$d")
    count=$(ls "$d" 2>/dev/null | wc -l)
    echo "  $name: $count"
done
echo ""
echo "=== Sample image ==="
f=$(ls /mnt/i/legacy256_overfit50/test/cezanne/*.jpg 2>/dev/null | head -1)
/home/xy/venvs/samam312/bin/python -c "from PIL import Image; img=Image.open('$f'); print(f'Size: {img.size}')"
