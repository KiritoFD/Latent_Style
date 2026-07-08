#!/bin/bash
# Try multiple mamba-ssm wheel URLs
set -e
source /root/samam_venv/bin/activate

echo "=== Try multiple wheel URLs ==="
WHEELS=(
    "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu121torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu118torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

SUCCESS=0
for URL in "${WHEELS[@]}"; do
    echo ""
    echo "Trying: $URL"
    rm -f /tmp/mamba.whl
    curl -L -s -o /tmp/mamba.whl "$URL" 2>&1 | tail -3
    SIZE=$(stat -c%s /tmp/mamba.whl 2>/dev/null || echo 0)
    echo "  size: $SIZE bytes"
    if [ "$SIZE" -gt 1000000 ]; then
        echo "  Looks like a valid wheel!"
        file /tmp/mamba.whl
        pip install /tmp/mamba.whl --no-build-isolation --no-deps 2>&1 | tail -10
        if python -c "import mamba_ssm" 2>/dev/null; then
            echo "  SUCCESS!"
            SUCCESS=1
            break
        fi
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "=== All URLs failed. Try via PyPI sdist with --no-build-isolation ==="
    # Use pip download with proper options
    cd /tmp
    rm -rf mamba_pypi
    mkdir -p mamba_pypi
    cd mamba_pypi
    # Use pip's --no-binary option to force source download
    pip download mamba-ssm==1.2.2 --no-deps --no-binary=:all: --no-build-isolation -d . 2>&1 | tail -10
    ls -la
fi

echo ""
echo "=== Verify ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python -c "import mamba_ssm; print('mamba_ssm:', mamba_ssm.__version__)" 2>&1
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('selective_scan_fn OK')" 2>&1
python -c "from PIL import Image; print('PIL OK')" 2>&1
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1
python -c "import tqdm; print('tqdm OK')" 2>&1

echo ""
echo "=== DONE ==="
