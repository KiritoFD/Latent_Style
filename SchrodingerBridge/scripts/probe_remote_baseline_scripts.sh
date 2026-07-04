#!/usr/bin/env bash
echo "=== List baseline_pipeline/scripts/ run_*.py ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/scripts/run_*.py 2>/dev/null
echo ""
echo "=== List baseline_pipeline/evaluation/ ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/evaluation/ 2>/dev/null
echo ""
echo "=== Check if legacy256 dataset already on remote ==="
find /mnt/i -maxdepth 4 -type d -iname "*legacy256*" 2>/dev/null | head -10
echo ""
echo "=== Check style_data/overfit50/test contents ==="
ls /mnt/i/Github/Latent_Style/style_data/overfit50/test/ 2>/dev/null
ls /mnt/i/Github/Latent_Style/style_data/overfit50/ 2>/dev/null
echo ""
echo "=== Check style_data/test (the real test set) ==="
ls /mnt/i/Github/Latent_Style/style_data/test/cezanne/ 2>/dev/null | head -3
echo "count: $(ls /mnt/i/Github/Latent_Style/style_data/test/cezanne/ 2>/dev/null | wc -l)"
echo ""
echo "=== Find adain/wct inference scripts ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -name "run_adain*" -o -name "run_wct*" 2>/dev/null | head -10
