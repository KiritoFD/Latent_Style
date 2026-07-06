#!/usr/bin/env bash
set -uo pipefail
echo "===512_pixel256 test==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/ 2>/dev/null
echo "===test dir structure==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/test 2>/dev/null | head -10
echo "===count per style==="
for s in /mnt/i/wikiart_distinct5_samam_512_pixel256/test/*/; do
    n=$(ls "$s" 2>/dev/null | wc -l)
    echo "$(basename $s): $n images"
done
echo "===legacy256_overfit50 test (training test_dir)==="
ls /mnt/i/legacy256_overfit50/test 2>/dev/null | head -5
for s in /mnt/i/legacy256_overfit50/test/*/; do
    n=$(ls "$s" 2>/dev/null | wc -l)
    echo "$(basename $s): $n images"
done
