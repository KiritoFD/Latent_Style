#!/usr/bin/env bash
echo "===== 1. SeeDream 750 dir structure ====="
SEEDREAM=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750
ls -la "$SEEDREAM" 2>/dev/null | head -20
echo "Image count: $(find "$SEEDREAM" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)"
echo "First 5 images:"
find "$SEEDREAM" -name "*.png" -o -name "*.jpg" 2>/dev/null | head -5
echo "Style dirs (if any):"
ls -d "$SEEDREAM"/*/ 2>/dev/null | head -10

echo ""
echo "===== 2. Existing baseline eval dir structure (for reference) ====="
echo "--- adain dir ---"
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/adain/ 2>/dev/null | head -10
echo "--- adain/images sample ---"
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/adain/images/ 2>/dev/null | head -5

echo ""
echo "===== 3. run_evaluation.py interface ====="
head -80 /mnt/i/Github/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py 2>/dev/null

echo ""
echo "===== 4. HF eval still running? ====="
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log
echo "Evaluated count: $(grep -c '"step":' "$EVAL_LOG" 2>/dev/null)"
echo "Last JSON:"
grep '"step":' "$EVAL_LOG" 2>/dev/null | tail -1
echo "Last log line:"
tail -1 "$EVAL_LOG" 2>/dev/null
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
