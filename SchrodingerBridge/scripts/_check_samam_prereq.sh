#!/bin/bash
source /root/samam_venv/bin/activate
echo "=== Check SaMam repo ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt 2>&1
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TEST/test_utils.py 2>&1
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/lightning_module/lightningmodel.py 2>&1

echo ""
echo "=== Check test set ==="
ls /mnt/i/datasets/wikiarts20_512_test/ 2>&1 | head -10
echo "--- 5 distinct5 styles ---"
for s in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    COUNT=$(ls /mnt/i/datasets/wikiarts20_512_test/$s/ 2>/dev/null | wc -l)
    echo "  $s: $COUNT images"
done

echo ""
echo "=== Check output dir ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/ 2>&1
echo "--- images dir ---"
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/ 2>&1 | head -5
echo "Total existing:"
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/*.png 2>/dev/null | wc -l
