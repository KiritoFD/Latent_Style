#!/bin/bash
echo "=== Check gen_image paths in metrics.csv ==="
head -5 /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/metrics.csv | cut -d',' -f4 | head -5
echo "=== Check if images exist elsewhere ==="
find /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/ -name "*.png" 2>/dev/null | head -5
echo "=== Check summary.json ==="
python3 -c "import json; d=json.load(open('/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/summary.json')); print('clip_style:', d.get('clip_style','N/A')); print('content_lpips:', d.get('content_lpips','N/A')); print('n_pairs:', d.get('n_pairs','N/A'))"
echo "=== Check other experiments ==="
find /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_film_formal/ -name "*.png" 2>/dev/null | head -3
find /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_lowswd_formal/ -name "*.png" 2>/dev/null | head -3
