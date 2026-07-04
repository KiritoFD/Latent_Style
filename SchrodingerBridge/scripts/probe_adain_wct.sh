#!/usr/bin/env bash
echo "=== Check adain repo ==="
ls /mnt/i/Github/Latent_Style/Related_Works/run_511/repos/adain/ 2>/dev/null | head -20
echo ""
echo "=== Check for adain decoder weights ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -name "*.pth" -path "*adain*" 2>/dev/null | head -10
find /mnt/i/Github/Latent_Style -maxdepth 5 -name "decoder*" -path "*adain*" 2>/dev/null | head -10
echo ""
echo "=== Check for wct scripts/weights ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -name "*wct*" 2>/dev/null | head -10
echo ""
echo "=== Check for VGG normalised weights ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -name "vgg_normalised*" 2>/dev/null | head -10
find /mnt/i/Github/Latent_Style -maxdepth 6 -name "vgg*19*" 2>/dev/null | head -10
echo ""
echo "=== Check SchrodingerBridge/tools for adain/wct ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ 2>/dev/null | grep -i -E "adain|wct" | head -10
echo ""
echo "=== Read run_adain_750.py inference section (lines 200-300) ==="
sed -n '200,300p' /mnt/i/Github/Latent_Style/Related_Works/run_511/launchers/run_adain_750.py 2>/dev/null
