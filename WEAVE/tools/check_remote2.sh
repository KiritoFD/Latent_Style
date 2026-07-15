#!/bin/bash
echo "=== intrinsic_v2 epoch_0008 ==="
ls -la /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/images/ 2>/dev/null | head -10
echo "=== metrics.csv sample ==="
head -3 /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/metrics.csv 2>/dev/null
echo "=== full_eval structure ==="
ls /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/ 2>/dev/null
echo "=== epoch_0008 structure ==="
ls -la /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/epoch_0008/ 2>/dev/null
