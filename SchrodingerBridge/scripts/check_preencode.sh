#!/usr/bin/env bash
echo "=== latent log (tail 20) ==="
tail -20 /mnt/i/exp_256_photo2art/_preencode_legacy256_latent.log 2>/dev/null
echo ""
echo "=== pixel log (tail 20) ==="
tail -20 /mnt/i/exp_256_photo2art/_preencode_legacy256_pixel.log 2>/dev/null
echo ""
echo "=== running processes ==="
ps -ef 2>/dev/null | grep -E "preencode|encode_image" | grep -v grep
echo ""
echo "=== latent output count ==="
ls /mnt/i/legacy256_overfit50_latent256/train/ 2>/dev/null
for s in cezanne Hayao monet photo vangogh; do
    n=$(ls /mnt/i/legacy256_overfit50_latent256/train/$s/*.pt 2>/dev/null | wc -l)
    echo "  $s: $n .pt"
done
echo ""
echo "=== pixel output count ==="
ls /mnt/i/legacy256_overfit50_pixel256/train/ 2>/dev/null
for s in cezanne Hayao monet photo vangogh; do
    n=$(ls /mnt/i/legacy256_overfit50_pixel256/train/$s/*.pt 2>/dev/null | wc -l)
    echo "  $s: $n .pt"
done
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null
