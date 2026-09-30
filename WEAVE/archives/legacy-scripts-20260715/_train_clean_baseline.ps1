$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Verify code is ASG-free
Write-Output "=== Verifying ASG-free codebase ==="
python -c @"
import sys
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from pathlib import Path
cfg = load_experiment_config(Path('configs/refactor_clean_baseline.json').resolve())
print('contract_family:', cfg.model.contract_family)
from spectral_bridge620 import SpectralODEBridge620
model = SpectralODEBridge620(cfg.model, cfg.bridge)
asg_count = sum(1 for blk in model.blocks if hasattr(blk, 'asg_proj'))
print(f'ASG modules found: {asg_count} (expected 0)')
assert asg_count == 0, 'ASG still present in code!'
print('PASS: codebase is ASG-free')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Output "FATAL: ASG verification failed"
    exit 1
}

# Start training
Write-Output "=== Starting clean baseline training (5 epochs, batch=24) ==="
$logFile = "C:\Users\Administrator\logs\train_clean_baseline.log"
python run.py --config configs\refactor_clean_baseline.json *>&1 | Tee-Object -FilePath $logFile

Write-Output "TRAIN_EXIT_CODE=$LASTEXITCODE"
