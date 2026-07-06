#!/usr/bin/env bash
set -uo pipefail
echo "===existing full_eval contents==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/ 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/ -type f 2>/dev/null | head -20
echo "===512_classview test structure==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/test/ 2>/dev/null
echo "---photo dir contents (first 5)---"
ls /mnt/i/wikiart_distinct5_samam_512_classview/test/photo/ 2>/dev/null | head -5
echo "---file extension counts---"
for s in /mnt/i/wikiart_distinct5_samam_512_classview/test/*/; do
    name=$(basename "$s")
    jpg=$(ls "$s"*.jpg 2>/dev/null | wc -l)
    png=$(ls "$s"*.png 2>/dev/null | wc -l)
    webp=$(ls "$s"*.webp 2>/dev/null | wc -l)
    echo "$name: jpg=$jpg png=$png webp=$webp"
done
echo "===check image size of 512_classview test==="
first_img=$(ls /mnt/i/wikiart_distinct5_samam_512_classview/test/photo/* 2>/dev/null | head -1)
file "$first_img" 2>/dev/null
