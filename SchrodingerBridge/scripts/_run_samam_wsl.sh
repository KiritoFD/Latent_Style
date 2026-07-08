#!/bin/bash
# Delete old 256-res SaMam images, then run WSL SaMam at 512-res
set -e
source /root/samam_venv/bin/activate

echo "=== STEP 1: Delete old 256-res SaMam images ==="
OLD_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images"
OLD_DONE="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/_DONE"
COUNT_BEFORE=$(ls "$OLD_DIR"/*.png 2>/dev/null | wc -l)
echo "  before: $COUNT_BEFORE png files"
rm -f "$OLD_DIR"/*.png
rm -f "$OLD_DONE"
COUNT_AFTER=$(ls "$OLD_DIR"/*.png 2>/dev/null | wc -l)
echo "  after: $COUNT_AFTER png files"
mkdir -p "$OLD_DIR"

echo ""
echo "=== STEP 2: Verify env ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "from PIL import Image; print('PIL OK')"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import tqdm; print('tqdm OK')"

echo ""
echo "=== STEP 3: Check SaMam repo + ckpt ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt 2>&1
ls /mnt/i/datasets/wikiarts20_512_test/ 2>&1 | head -5

echo ""
echo "=== STEP 4: Run SaMam Random5 generation ==="
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/_gen_samam_random5_wsl.py 2>&1

echo ""
echo "=== DONE ==="
date
