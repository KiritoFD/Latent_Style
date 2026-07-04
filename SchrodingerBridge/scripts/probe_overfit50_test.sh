#!/usr/bin/env bash
echo "=== style_data/overfit50/test/cezanne count ==="
ls /mnt/i/Github/Latent_Style/style_data/overfit50/test/cezanne/ 2>/dev/null | wc -l
echo "sample files:"
ls /mnt/i/Github/Latent_Style/style_data/overfit50/test/cezanne/ 2>/dev/null | head -3
echo ""
echo "=== Sample image size ==="
f=$(ls /mnt/i/Github/Latent_Style/style_data/overfit50/test/cezanne/*.jpg 2>/dev/null | head -1)
if [ -n "$f" ]; then
    /home/xy/venvs/samam312/bin/python -c "from PIL import Image; img=Image.open('$f'); print(f'Size: {img.size}')"
fi
echo ""
echo "=== style_data/test/cezanne sample size ==="
f=$(ls /mnt/i/Github/Latent_Style/style_data/test/cezanne/*.jpg 2>/dev/null | head -1)
if [ -n "$f" ]; then
    /home/xy/venvs/samam312/bin/python -c "from PIL import Image; img=Image.open('$f'); print(f'Size: {img.size}')"
fi
echo ""
echo "=== adain_750.py head ==="
head -40 /mnt/i/Github/Latent_Style/Related_Works/run_511/launchers/run_adain_750.py 2>/dev/null
