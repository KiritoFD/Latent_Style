#!/usr/bin/env bash
echo "===== 1. exp/baseline_v2/eval/samam/summary.json ====="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/summary.json 2>/dev/null

echo ""
echo "===== 2. exp/baseline_v2/eval/samam/metrics.csv (head+tail) ====="
head -3 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/metrics.csv 2>/dev/null
echo "..."
tail -3 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/metrics.csv 2>/dev/null
echo "Row count: $(wc -l < /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam/metrics.csv 2>/dev/null)"

echo ""
echo "===== 3. samam_256_faithful_p8_remote structure ====="
DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_256_faithful_p8_remote
ls -la "$DIR" 2>/dev/null | head -20
echo "--- b1_sph35_20260522_050523 ---"
ls -la "$DIR/b1_sph35_20260522_050523" 2>/dev/null | head -20

echo ""
echo "===== 4. samam_256_faithful train script ====="
find "$DIR" -name "*.sh" 2>/dev/null | head -5
for f in $(find "$DIR" -name "*.sh" 2>/dev/null | head -2); do
    echo "--- $f ---"
    cat "$f"
done

echo ""
echo "===== 5. h03_step0105/x8/metrics.csv (the 0.722169 hit) ====="
METRICS="$DIR/b1_sph35_20260522_050523/h03_step0105/x8/metrics.csv"
head -3 "$METRICS" 2>/dev/null
echo "..."
echo "Row count: $(wc -l < "$METRICS" 2>/dev/null)"
echo "--- last row mean clip_style ---"
tail -1 "$METRICS" 2>/dev/null | cut -d',' -f1-8

echo ""
echo "===== 6. h03_step0105 summary ====="
SUMM="$DIR/b1_sph35_20260522_050523/h03_step0105/x8/summary.json"
cat "$SUMM" 2>/dev/null | head -40

echo ""
echo "===== 7. 20K train final stats ====="
NEWDIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
grep -E "TRAIN_DONE|WALL_SECONDS|END=" "$NEWDIR/train_resume_20k.log" 2>/dev/null | tail -5
echo "Checkpoint count: $(ls $NEWDIR/step_checkpoints/step-step=*.ckpt 2>/dev/null | wc -l)"

echo ""
echo "===== 8. done ====="
