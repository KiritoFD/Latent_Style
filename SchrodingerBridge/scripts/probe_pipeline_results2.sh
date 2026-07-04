#!/usr/bin/env bash
echo "=== style_data/overfit50 ==="
ls /mnt/i/Github/Latent_Style/style_data/overfit50/ 2>/dev/null
echo ""
echo "=== style_data/overfit50/test ==="
ls /mnt/i/Github/Latent_Style/style_data/overfit50/test/ 2>/dev/null
echo ""
echo "=== style_data/test ==="
ls /mnt/i/Github/Latent_Style/style_data/test/ 2>/dev/null | head -10
echo ""
echo "=== style_data/train ==="
ls /mnt/i/Github/Latent_Style/style_data/train/ 2>/dev/null | head -10
echo ""
echo "=== Search for any baseline output dirs with _to_photo _to_monet _to_cezanne ==="
find /mnt/i/Github/Latent_Style/Related_Works -maxdepth 5 -type d -name "*to_photo*" -o -name "*to_monet*" -o -name "*to_cezanne*" 2>/dev/null | head -10
echo ""
echo "=== Check Related_Works/results/metrics_summary ==="
ls /mnt/i/Github/Latent_Style/Related_Works/results/metrics_summary/ 2>/dev/null | head -20
echo ""
echo "=== SchrodingerBridge/exp/baseline_v2/eval/seedream images count ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l
echo ""
echo "=== SchrodingerBridge/exp/baseline_v2/images dirs ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/ 2>/dev/null
