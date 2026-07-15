#!/usr/bin/env bash
echo "===== 1. samam_256_faithful_p8_remote dir ====="
DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_256_faithful_p8_remote
ls -la "$DIR" 2>/dev/null | head -30

echo ""
echo "===== 2. Its train.log (config + wall) ====="
if [ -f "$DIR/train.log" ]; then
    head -30 "$DIR/train.log" | tr '\r' '\n' | head -30
    echo "..."
    tail -10 "$DIR/train.log" | tr '\r' '\n' | tail -10
fi

echo ""
echo "===== 3. Its run script ====="
ls "$DIR"/*.sh 2>/dev/null
for f in "$DIR"/*.sh; do
    echo "--- $f ---"
    cat "$f" 2>/dev/null
done

echo ""
echo "===== 4. Its eval results ====="
find "$DIR" -name "summary.json" -o -name "metrics.csv" -o -name "*curve*" 2>/dev/null | head -10
for f in $(find "$DIR" -name "summary.json" 2>/dev/null | head -3); do
    echo "--- $f ---"
    head -50 "$f"
done

echo ""
echo "===== 5. exp/baseline_v2/eval/samam/ on I drive ====="
SAMAM_EVAL=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/samam
ls -la "$SAMAM_EVAL" 2>/dev/null
echo "--- summary.json ---"
cat "$SAMAM_EVAL/summary.json" 2>/dev/null
echo ""
echo "--- metrics.csv head ---"
head -5 "$SAMAM_EVAL/metrics.csv" 2>/dev/null
echo "--- metrics.csv tail ---"
tail -5 "$SAMAM_EVAL/metrics.csv" 2>/dev/null

echo ""
echo "===== 6. Search 0.722169 in samam_256_faithful ====="
grep -rl "0.722169\|0.7222\|0.328176" "$DIR" 2>/dev/null | head -10

echo ""
echo "===== 7. Search 0.722169 in exp/baseline_v2 ====="
grep -rl "0.722169\|0.7222" /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/ 2>/dev/null | head -10

echo ""
echo "===== 8. remote_master_baseline_v2.py samam section ====="
grep -n "samam\|0.7222\|samam_256" /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/remote_master_baseline_v2.py 2>/dev/null | head -20

echo ""
echo "===== 9. 7k eval (open_clip) results - the NEW 20k train output ====="
NEW_CKPT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
echo "Checkpoint count:"
ls "$NEW_CKPT"/step_checkpoints/step-step=*.ckpt 2>/dev/null | wc -l
echo "Last 3 checkpoints:"
ls "$NEW_CKPT"/step_checkpoints/step-step=*.ckpt 2>/dev/null | tail -3
echo "HF eval output exists?"
ls -la "$NEW_CKPT"/curve_eval_hf_750/ 2>/dev/null | head -5
echo "Old open_clip eval:"
ls "$NEW_CKPT"/curve_eval_30src/curve_metrics.csv 2>/dev/null && echo "exists"

echo ""
echo "===== 10. done ====="
