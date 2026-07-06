#!/bin/bash
# Check WSL SaMam environment
set -e

echo "=== WSL Python ==="
which python3
python3 --version

echo ""
echo "=== mamba_ssm ==="
python3 -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)"

echo ""
echo "=== torch ==="
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python3 -c "import torch; print('VRAM total:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB') if torch.cuda.is_available() else None"

echo ""
echo "=== Other deps ==="
python3 -c "from PIL import Image; print('PIL OK')"
python3 -c "from torchvision.utils import save_image; print('torchvision OK')"

echo ""
echo "=== SaMam repo path ==="
ls -la /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN/final_model.ckpt
ls /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TEST/test_utils.py 2>&1

echo ""
echo "=== Test data ==="
ls /mnt/i/datasets/wikiarts20_512_test/ | head -10
echo "Style count: $(ls /mnt/i/datasets/wikiarts20_512_test/ | wc -l)"

echo ""
echo "=== Output dir ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/ 2>&1 | head -3
echo "Existing count: $(ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images/ 2>/dev/null | wc -l)"
