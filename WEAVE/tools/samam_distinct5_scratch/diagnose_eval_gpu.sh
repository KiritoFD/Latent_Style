#!/usr/bin/env bash
echo "===== 1. Find eval_samam_checkpoint_curve.py ====="
find /mnt/i/Github/Latent_Style -name "eval_samam_checkpoint_curve.py" 2>/dev/null | head -5

echo ""
echo "===== 2. Check device/cuda usage in eval script ====="
EVAL_SCRIPT=$(find /mnt/i/Github/Latent_Style -name "eval_samam_checkpoint_curve.py" 2>/dev/null | head -1)
echo "Script: $EVAL_SCRIPT"
echo "--- grep cuda/device/cpu ---"
grep -n "cuda\|device\|cpu\|\.to(\|batch_size\|DataLoader\|num_workers" "$EVAL_SCRIPT" 2>/dev/null | head -40

echo ""
echo "===== 3. Check CLIP/LPIPS model loading ====="
grep -n "CLIPModel\|clip_model\|lpips\|LPIPS\|open_clip\|transformers" "$EVAL_SCRIPT" 2>/dev/null | head -20

echo ""
echo "===== 4. Check run_evaluation.py (the actual eval) ====="
RUN_EVAL=/mnt/i/Github/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py
echo "--- grep cuda/device ---"
grep -n "cuda\|device\|\.to(\|batch_size" "$RUN_EVAL" 2>/dev/null | head -20

echo ""
echo "===== 5. Check current GPU memory user (PID 5425) ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "--- py-spy dump (if installed) ---"
py-spy dump --pid 5425 2>/dev/null | head -30 || echo "(py-spy not installed)"

echo ""
echo "===== 6. Training GPU usage (for comparison) ====="
grep -i "gpu\|cuda\|device" /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train_resume_20k.log 2>/dev/null | head -5
