#!/bin/bash
echo "=== I盘根目录文件 ==="
ls -la /mnt/i/ 2>/dev/null | head -20

echo ""
echo "=== Github 目录占用 ==="
du -sh /mnt/i/Github 2>/dev/null

echo ""
echo "=== 子目录一级占用 ==="
for d in /mnt/i/*/; do
    if [ -d "$d" ]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "$size | $d"
    fi
done | sort -rh | head -15

echo ""
echo "=== 可用空间 ==="
df -h /mnt/i 2>/dev/null
