#!/bin/bash
echo "=== I盘空间占用 Top 20 ==="
du -h -d 1 /mnt/i 2>/dev/null | sort -rh | head -20

echo ""
echo "=== SchrodingerBridge 目录占用 ==="
du -h -d 1 /mnt/i/Github/Latent_Style/SchrodingerBridge 2>/dev/null | sort -rh | head -20

echo ""
echo "=== exp_ablation_620 占用 ==="
du -h -d 1 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 2>/dev/null | sort -rh | head -10

echo ""
echo "=== eval_cache 占用 ==="
du -h -d 1 /mnt/i/Github/Latent_Style/eval_cache 2>/dev/null | sort -rh | head -10

echo ""
echo "=== Dataset 占用 ==="
du -h -d 1 /mnt/i/Dataset 2>/dev/null | sort -rh | head -10 || echo "No /mnt/i/Dataset"

echo ""
echo "=== 大文件搜索 (>5GB) ==="
find /mnt/i -type f -size +5G 2>/dev/null | head -20
