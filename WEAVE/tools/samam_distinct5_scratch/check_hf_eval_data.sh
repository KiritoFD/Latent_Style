#!/usr/bin/env bash
echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== EVAL LOG TAIL ==="
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log
tail -30 "$EVAL_LOG" 2>/dev/null

echo ""
echo "=== EVALUATED COUNT ==="
grep -c '"step":' "$EVAL_LOG" 2>/dev/null || echo 0

echo ""
echo "=== HF curve_metrics.csv (if exists) ==="
HF_CSV=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750/curve_metrics.csv
if [ -f "$HF_CSV" ]; then
    echo "File exists, rows: $(wc -l < "$HF_CSV")"
    echo "--- full content ---"
    cat "$HF_CSV"
else
    echo "(no CSV yet, checking individual step dirs)"
    HF_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750
    ls "$HF_DIR" 2>/dev/null | head -20
    echo ""
    echo "--- check summary.json in each step ---"
    for d in "$HF_DIR"/step_*/; do
        step=$(basename "$d")
        if [ -f "$d/summary.json" ]; then
            clip=$(python3 -c "import json; d=json.load(open('$d/summary.json')); print(f\"clip={d.get('clip_style','?')}, lpips={d.get('content_lpips','?')}\")" 2>/dev/null)
            echo "$step: $clip"
        fi
    done 2>/dev/null | head -30
fi

echo ""
echo "=== OLD open_clip CSV (for reference, 28 ckpts 250-7000) ==="
OLD_CSV=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_30src/curve_metrics.csv
if [ -f "$OLD_CSV" ]; then
    head -1 "$OLD_CSV"
    tail -20 "$OLD_CSV"
fi

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
