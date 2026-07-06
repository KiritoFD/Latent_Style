#!/usr/bin/env bash
set -uo pipefail
echo "===overfit50 test dir (current config)==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/style_data/overfit50/ 2>/dev/null | head -10
echo "===overfit50 sample style==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/style_data/overfit50/Early_Renaissance/ 2>/dev/null | head -5
echo "===count per style==="
for s in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    cnt=$(ls /mnt/i/Github/Latent_Style/SchrodingerBridge/style_data/overfit50/$s/ 2>/dev/null | wc -l)
    echo "  $s: $cnt"
done

echo ""
echo "===wikiart_distinct5_samam_512_classview/test (project constraint)==="
ls /mnt/i/wikiart_distinct5_samam_512_classview/test/ 2>/dev/null | head -10
for s in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    cnt=$(ls /mnt/i/wikiart_distinct5_samam_512_classview/test/$s/ 2>/dev/null | wc -l)
    echo "  $s: $cnt"
done

echo ""
echo "===DA01 ckpt details==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/epoch_0003.pt 2>/dev/null
echo "===DA01 config save_dir==="
grep -E "save_dir|test_image_dir" /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/config.json 2>/dev/null | head -5
