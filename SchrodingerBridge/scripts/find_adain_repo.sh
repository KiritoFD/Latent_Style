#!/usr/bin/env bash
echo "=== pytorch-AdaIN repo ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN/ 2>/dev/null
echo ""
echo "=== Find decoder.py ==="
find /mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN -name "*.py" 2>/dev/null
echo ""
echo "=== Find decoder.pth ==="
find /mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN -name "*.pth" 2>/dev/null
echo ""
echo "=== run_511/repos/adain ==="
ls /mnt/i/Github/Latent_Style/Related_Works/run_511/repos/adain/ 2>/dev/null
echo ""
echo "=== adain_net.py decoder definition ==="
grep -n "class\|decoder\|Decoder" /mnt/i/Github/Latent_Style/Related_Works/run_511/repos/adain/adain_net.py 2>/dev/null | head -20
