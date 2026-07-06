#!/bin/bash
# Check SaMam env in WSL

echo "=== SaMam repo ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/ | head -10

echo ""
echo "=== CKPT ==="
ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt 2>&1 || echo "NOT FOUND"

echo ""
echo "=== Python ==="
which python python3 2>&1
python3 --version 2>&1 || python --version 2>&1

echo ""
echo "=== mamba_ssm check ==="
python3 -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1 || \
  python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)" 2>&1

echo ""
echo "=== torch check ==="
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1 || \
  python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1

echo ""
echo "=== TEST module check ==="
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TEST/ 2>&1 | head -10

echo ""
echo "=== wiki20distinct5 test data ==="
ls /mnt/i/datasets/wikiarts20_512_test/ 2>&1 | head -10
