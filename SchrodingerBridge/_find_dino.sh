#!/bin/bash
# 查找 DINO cache 和 L9 配置
echo "=== L9 init configs ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/625_fc_sb/from_scratch_win/init_configs/ 2>/dev/null
echo ""
echo "=== Search dino cache ==="
find /mnt/i/Github/Latent_Style/eval_cache -name "*dinov2*" 2>/dev/null
find /mnt/i/Github/Latent_Style/eval_cache -name "*dino*cache*" 2>/dev/null
echo ""
echo "=== L9 I7 config dino_cache_path ==="
grep -i "dino_cache" /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/625_fc_sb/from_scratch_win/init_configs/I7.json 2>/dev/null
echo ""
echo "=== Existing successful exp configs ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name "*.json" -path "*from_scratch*" 2>/dev/null | head -5
echo ""
echo "=== Search any dino cache on I drive ==="
find /mnt/i -maxdepth 5 -name "*dinov2*train*cache*" 2>/dev/null | head -5
