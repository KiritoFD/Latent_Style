#!/usr/bin/env bash
echo "=== Find 256 dataset dirs on /mnt/i ==="
find /mnt/i -maxdepth 3 -type d -iname "*256*" 2>/dev/null | head -30
echo ""
echo "=== Check /mnt/i for photo2art-style 5-class dataset ==="
find /mnt/i -maxdepth 4 -type d \( -iname "cezanne" -o -iname "hayao" -o -iname "monet" -o -iname "vangogh" \) 2>/dev/null | head -20
echo ""
echo "=== Check /mnt/i/wikiart_distinct5_samam_512_pixel256 ==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/ 2>/dev/null
echo ""
echo "=== Check test subdirs ==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/test 2>/dev/null
echo ""
echo "=== Check distinct5 structure ==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/ 2>/dev/null
ls /mnt/i/wikiart_distinct5_samam_512_classview/test 2>/dev/null
