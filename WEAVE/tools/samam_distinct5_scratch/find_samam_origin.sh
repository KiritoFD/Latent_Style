#!/usr/bin/env bash
echo "=== All samam dirs in I:/Github/Latent_Style/Related_Works/baseline_pipeline/results/ ==="
ls -dt /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_* 2>/dev/null | head -30

echo ""
echo "=== Search for batch=4 or 256 in samam dir names ==="
ls -d /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_*256* 2>/dev/null
ls -d /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_*b4* 2>/dev/null
ls -d /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_*b2* 2>/dev/null

echo ""
echo "=== Search for 0.7221691230138143 in all samam eval results ==="
grep -rl "0.7221691230138143\|0.7222\|0.3281765048" /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_* 2>/dev/null | head -20

echo ""
echo "=== exp/baseline_v2/eval/samam/ on I drive ==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/ 2>/dev/null
echo "--- summary.json (if exists) ---"
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/summary.json 2>/dev/null | head -50
echo "--- metrics.csv tail (if exists) ---"
tail -5 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/metrics.csv 2>/dev/null

echo ""
echo "=== Search 0.7222 in exp/baseline_v2 ==="
grep -rl "0.7221691230138143\|0.7222" /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/ 2>/dev/null | head -10

echo ""
echo "=== All SaMam train scripts showing batch/resolution ==="
for d in /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_*/; do
    name=$(basename "$d")
    # Look for train log or run script
    tlog="$d/train.log"
    rsh=$(ls "$d"/*.sh 2>/dev/null | head -1)
    cfg=""
    if [ -f "$rsh" ]; then
        cfg=$(grep -oE 'batch-size [0-9]+|train-image-size [0-9]+|iterations [0-9]+' "$rsh" 2>/dev/null | tr '\n' ' ')
    fi
    if [ -z "$cfg" ] && [ -f "$tlog" ]; then
        cfg=$(grep -oE 'batch.size.[0-9]+|image.size.[0-9]+|iterations.[0-9]+' "$tlog" 2>/dev/null | head -3 | tr '\n' ' ')
    fi
    echo "$name | $cfg"
done 2>/dev/null | head -30
