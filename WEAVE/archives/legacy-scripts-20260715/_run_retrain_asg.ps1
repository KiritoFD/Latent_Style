$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "src"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Verify run.py fix works (dry run -- just check imports)
Write-Output "=== Verifying run.py fix ==="
python -c "import sys; sys.path.insert(0, 'src'); import importlib.util; spec = importlib.util.spec_from_file_location('run_check', 'src/run.py'); m = importlib.util.module_from_spec(spec); print('src/run.py loads OK:', hasattr(m, 'main'))"

# Verify ASG is active in the config
Write-Output "=== Verifying ASG activation ==="
python -c @"
import sys, json
sys.path.insert(0, 'src')
from config_schema import load_experiment_config
from pathlib import Path
cfg = load_experiment_config(Path('configs/t1_asg_5ep.json').resolve())
print('adaptive_style_gate:', cfg.model.adaptive_style_gate)
print('contract_family:', cfg.model.contract_family)
"@

# Start retraining t1_asg_5ep with fixed code
Write-Output "=== Starting retrain t1_asg_5ep ==="
$logFile = "C:\Users\Administrator\logs\retrain_asg_5ep.log"
python run.py --config configs\t1_asg_5ep.json *>&1 | Tee-Object -FilePath $logFile

Write-Output "EXIT_CODE=$LASTEXITCODE"
