#!/usr/bin/env bash
echo "=== legacy256_overfit50/train 文件数 ==="
cnt=$(ls /mnt/i/legacy256_overfit50/train 2>/dev/null)
echo "$cnt"
echo ""
for s in cezanne Hayao monet photo vangogh; do
    n=$(ls /mnt/i/legacy256_overfit50/train/$s 2>/dev/null | wc -l)
    echo "$s: $n"
done
echo ""
echo "=== legacy256_overfit50/test 文件数 ==="
for s in cezanne Hayao monet photo vangogh; do
    n=$(ls /mnt/i/legacy256_overfit50/test/$s 2>/dev/null | wc -l)
    echo "$s: $n"
done
