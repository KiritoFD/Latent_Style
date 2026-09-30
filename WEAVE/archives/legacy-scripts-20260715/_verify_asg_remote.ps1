$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Verify ASG activation in the active contract
Write-Output "=== Verifying ASG in active SpectralODEBridge620 ==="
python -c @"
import sys
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from pathlib import Path
cfg = load_experiment_config(Path('configs/t1_asg_5ep.json').resolve())
print('adaptive_style_gate:', cfg.model.adaptive_style_gate)
print('contract_family:', cfg.model.contract_family)
from spectral_bridge620 import SpectralODEBridge620
import torch
model = SpectralODEBridge620(cfg.model, num_styles=5)
print('model.adaptive_style_gate:', model.adaptive_style_gate)
for i, blk in enumerate(model.blocks):
    has_asg = hasattr(blk, 'asg_proj')
    print(f'block {i}: asg_proj={has_asg}')
    if has_asg:
        w = blk.asg_proj.weight
        print(f'  asg_proj.weight shape={tuple(w.shape)}, zero={bool((w.abs().sum()==0))}')
"@

# Rename old checkpoint dir
Write-Output "=== Cleaning existing checkpoint dir ==="
if (Test-Path "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep") {
    Rename-Item "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep" "t1_asg_5ep_old_noasg" -Force -ErrorAction SilentlyContinue
    Write-Output "Renamed old checkpoint dir to t1_asg_5ep_old_noasg"
} else {
    Write-Output "No existing checkpoint dir"
}

# Start retraining
Write-Output "=== Starting retrain t1_asg_5ep (5 epochs, batch=24) ==="
$logFile = "C:\Users\Administrator\logs\retrain_asg_5ep.log"
python run.py --config configs\t1_asg_5ep.json *>&1 | Tee-Object -FilePath $logFile

Write-Output "EXIT_CODE=$LASTEXITCODE"
