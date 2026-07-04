# Verify _628_infer_ablation_batch.py syntax and copy to remote
$src = 'I:/Github/Latent_Style/SchrodingerBridge/_628_infer_ablation_batch.py'
$python = 'C:\Progra~1\Python312\python.exe'

if (-not (Test-Path $src)) {
    Write-Host "ERROR: file not found: $src"
    exit 1
}

# Syntax check via ast.parse
$checker = @"
import ast, sys
try:
    with open(r'$src', encoding='utf-8') as f:
        ast.parse(f.read())
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax ERROR: {e}')
    sys.exit(1)
"@

& $python -c $checker
if ($LASTEXITCODE -ne 0) {
    Write-Host "Syntax check failed"
    exit 1
}

# List ablations
Write-Host "`n=== Listing ablations ==="
& $python $src --list

# Check batch runner is still running
Write-Host "`n=== Batch runner status ==="
$pidFile = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_runner.pid'
if (Test-Path $pidFile) {
    $bpid = (Get-Content $pidFile -Raw).Trim()
    $p = Get-Process -Id $bpid -ErrorAction SilentlyContinue
    if ($p -and -not $p.HasExited) {
        Write-Host "Batch runner PID=$bpid RUNNING (CPU=$($p.CPU) WS=$([math]::Round($p.WorkingSet64/1MB,1))MB)"
    } else {
        Write-Host "Batch runner PID=$bpid EXITED"
    }
}

# Check batch log progress
Write-Host "`n=== batch_log.txt tail ==="
$batchLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_log.txt'
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 8
}

# Check pending count
Write-Host "`n=== Progress check ==="
$cfgDir = 'I:/Github/Latent_Style/SchrodingerBridge/configs/ablations/628_destructive'
$expDir = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive'
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0
$pending = 0
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) { $done++ } else { $pending++ }
}
Write-Host "Done: $done / $($allConfigs.Count) | Pending: $pending"

Write-Host "`n=== nvidia-smi ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
