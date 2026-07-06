#!/usr/bin/env bash
set -uo pipefail
echo "===existing pixel256 eval outputs==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art -name "summary.json" 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art -name "full_eval" -type d 2>/dev/null
echo "===pixel256 dirs structure==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/ 2>/dev/null
echo "===512_pixel256 train structure (sample)==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/train 2>/dev/null | head -10
echo "===check if 512_classview test images are 512x512==="
file /mnt/i/wikiart_distinct5_samam_512_classview/test/photo/*.jpg 2>/dev/null | head -2
file /mnt/i/wikiart_distinct5_samam_512_classview/test/photo/*.png 2>/dev/null | head -2
echo "===check legacy256_overfit50 test image size==="
file /mnt/i/legacy256_overfit50/test/photo/*.jpg 2>/dev/null | head -2
file /mnt/i/legacy256_overfit50/test/photo/*.png 2>/dev/null | head -2
echo "===existing 256 test dirs==="
find /mnt/i -maxdepth 3 -type d -name "test" -path "*256*" 2>/dev/null | head -10
