#!/bin/bash
# Run SaMam Random5 generation in WSL with real mamba_ssm
# This script runs from /mnt/c/Users/Administrator/
set -e
source /root/samam_venv/bin/activate

LOG=/tmp/samam_gen.log

echo "=== STEP 1: Delete old SaMam images ===" | tee $LOG
OLD_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images"
OLD_DONE="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/_DONE"
COUNT_BEFORE=$(ls "$OLD_DIR"/*.png 2>/dev/null | wc -l)
echo "  before: $COUNT_BEFORE png files" | tee -a $LOG
rm -f "$OLD_DIR"/*.png 2>/dev/null || true
rm -f "$OLD_DONE" 2>/dev/null || true
mkdir -p "$OLD_DIR"
COUNT_AFTER=$(ls "$OLD_DIR"/*.png 2>/dev/null | wc -l)
echo "  after: $COUNT_AFTER png files" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== STEP 2: Verify env ===" | tee -a $LOG
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1 | tee -a $LOG
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1 | tee -a $LOG
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>&1 | tee -a $LOG
python -c "from PIL import Image; print('PIL OK')" 2>&1 | tee -a $LOG
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== STEP 3: Check SaMam repo + ckpt ===" | tee -a $LOG
ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt 2>&1 | tee -a $LOG
ls /mnt/i/datasets/wikiarts20_512_test/ 2>&1 | head -10 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== STEP 4: Run SaMam Random5 generation ===" | tee -a $LOG
echo "START=$(date)" | tee -a $LOG
python /mnt/c/Users/Administrator/_gen_samam_random5_wsl.py 2>&1 | tee -a $LOG
RC=$?
echo "END=$(date)" | tee -a $LOG
echo "rc=$RC" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== STEP 5: Final count ===" | tee -a $LOG
ls "$OLD_DIR"/*.png 2>/dev/null | wc -l | tee -a $LOG
ls "$OLD_DONE" 2>/dev/null | tee -a $LOG

exit $RC
