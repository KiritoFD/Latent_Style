#!/usr/bin/env bash
echo "=== WSL ALIVE ==="
date -Iseconds

echo "=== CHECKPOINTS ==="
CKPT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints
ls "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l
echo "First/last few:"
ls "$CKPT_DIR" 2>/dev/null | head -5
ls "$CKPT_DIR" 2>/dev/null | tail -5

echo "=== EXISTING EVAL OUTPUTS ==="
ls -la /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/ 2>/dev/null | grep -E "curve_eval|eval.log"

echo "=== HF CLIP CACHE ==="
find /mnt/i/Github/Latent_Style/eval_cache/hf -maxdepth 3 -name "*clip*" 2>/dev/null | head -10
ls -la /mnt/i/Github/Latent_Style/eval_cache/hf/hub/ 2>/dev/null | head -15

echo "=== PYTHON ENV ==="
source /home/xy/venvs/samam312/bin/activate
python -c "import transformers; print('transformers', transformers.__version__)" 2>&1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
python -c "import lpips; print('lpips ok')" 2>&1

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo "=== DONE ==="
