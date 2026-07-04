#!/usr/bin/env bash
echo "=== samam_256 sample filenames ==="
ls /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/images/ 2>/dev/null | head -5
echo ""
echo "=== samam_256 image count ==="
ls /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/images/ 2>/dev/null | wc -l
echo ""
echo "=== samam_256 config ==="
find /mnt/i/Github/Latent_Style/exp_samam/training -name "config.json" -path "*256*" 2>/dev/null | head -3
echo ""
echo "=== Check samam_256 training config style_subdirs ==="
find /mnt/i/Github/Latent_Style/exp_samam/training -name "*.yaml" -o -name "*.yml" -o -name "config.json" 2>/dev/null | head -5
echo ""
echo "=== Search for any 256 photo2art baseline outputs ==="
find /mnt/i -maxdepth 7 -name "*_to_cz*.png" -o -name "*_to_Hayao*.png" -o -name "*_to_monet*.png" -o -name "*_to_vangogh*.png" 2>/dev/null | head -10
echo ""
echo "=== legacy256 dataset on remote? ==="
find /mnt/i -maxdepth 6 -type d -iname "*legacy*256*" 2>/dev/null | head -10
echo ""
echo "=== Check distinct5 256 test dir used by current 256 baselines ==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/test 2>/dev/null | head -10
