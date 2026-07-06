# Test Windows python with one ablation config (smoke test)
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"
$CONFIG = "$REPO\configs\abl512_X06_no_spectral_ode.json"
$env:PYTHONPATH = "$REPO\src"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "$REPO\exp\eval_cache\hf"

Set-Location $REPO
Write-Host "=== Windows Python smoke test ==="
Write-Host "Python: $PYTHON"
Write-Host "Config: $CONFIG"
Write-Host "Repo:   $REPO"
Write-Host ""

Write-Host "=== Python version and torch check ==="
& $PYTHON -c "import sys; print('Python:', sys.version); import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
Write-Host ""

Write-Host "=== Config load test ==="
& $PYTHON -c "import json; c=json.load(open(r'$CONFIG')); print('Config loaded OK'); print('data_root:', c['data']['data_root']); print('test_image_dir:', c['training']['test_image_dir'])"
Write-Host ""

Write-Host "=== Dataset path existence check ==="
& $PYTHON -c "import os; p=r'I:\datasets\wikiart_distinct5_samam_512_latents_ema\train'; print('Train dir exists:', os.path.isdir(p)); p2=r'I:\datasets\wikiart_distinct5_samam_512_classview\test'; print('Test dir exists:', os.path.isdir(p2))"
Write-Host ""

Write-Host "=== DONE ==="
