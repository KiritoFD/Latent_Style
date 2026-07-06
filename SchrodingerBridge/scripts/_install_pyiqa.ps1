# Install pyiqa + addict + scikit-image from local whls (no internet needed)
$PYTHON = "C:\Program Files\Python312\python.exe"
$WHL_DIR = "I:\Github\Latent_Style\SchrodingerBridge\scripts"

Write-Host "=== Installing pyiqa dependencies from local whls ==="

# Install addict (pure python, no deps)
Write-Host "[1/3] Installing addict..."
& $PYTHON -m pip install --no-deps --no-index --find-links $WHL_DIR addict 2>&1 | Select-Object -Last 5

# Install scikit-image with --no-deps (we only need basic functions; pyiqa MUSIQ may not need full skimage)
Write-Host "[2/3] Installing scikit-image (no-deps)..."
& $PYTHON -m pip install --no-deps --no-index --find-links $WHL_DIR scikit-image 2>&1 | Select-Object -Last 5

# Install pyiqa with --no-deps
Write-Host "[3/3] Installing pyiqa (no-deps)..."
& $PYTHON -m pip install --no-deps --no-index --find-links $WHL_DIR pyiqa 2>&1 | Select-Object -Last 5

# Test import
Write-Host ""
Write-Host "=== Testing pyiqa import ==="
& $PYTHON -c "import pyiqa; print('pyiqa version:', pyiqa.__version__); print('Available metrics sample:', [m for m in dir(pyiqa) if 'musiq' in m.lower()][:5])"

# Test MUSIQ creation (may need weights download - will fail if weights not cached)
Write-Host ""
Write-Host "=== Testing MUSIQ metric creation ==="
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
& $PYTHON -c @"
import os
os.environ['TORCH_HOME'] = r'C:\Users\Administrator\.cache\torch'
import pyiqa
try:
    m = pyiqa.create_metric('musiq_koniq', device='cpu')
    print('MUSIQ KONIQ created OK')
except Exception as e:
    print(f'MUSIQ create failed: {e}')
    # Try with cached weights
    import torch
    print('Torch hub cache:', os.path.exists(os.path.join(os.environ.get('TORCH_HOME',''), 'hub', 'pyiqa')))
"@

# Check MUSIQ weights location
Write-Host ""
Write-Host "=== MUSIQ weights check ==="
$musiqWeight = "C:\Users\Administrator\musiq_koniq_ckpt-e95806b9.pth"
$torchHubDir = "C:\Users\Administrator\.cache\torch\hub\pyiqa"
$torchHubCheckpoints = "C:\Users\Administrator\.cache\torch\hub\checkpoints"
if (Test-Path $musiqWeight) {
    Write-Host "MUSIQ weight found: $musiqWeight"
    # Place it where pyiqa expects it
    if (-not (Test-Path $torchHubDir)) {
        New-Item -ItemType Directory -Force -Path $torchHubDir | Out-Null
    }
    if (-not (Test-Path $torchHubCheckpoints)) {
        New-Item -ItemType Directory -Force -Path $torchHubCheckpoints | Out-Null
    }
    Copy-Item $musiqWeight -Destination $torchHubDir -Force
    Copy-Item $musiqWeight -Destination $torchHubCheckpoints -Force
    Write-Host "Copied to: $torchHubDir"
    Write-Host "Copied to: $torchHubCheckpoints"
}

Write-Host ""
Write-Host "=== DONE ==="
