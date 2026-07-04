# Check batch runner recovery status after waiting
Start-Sleep -Seconds 480

$root = 'I:/Github/Latent_Style/SchrodingerBridge'

Write-Host "=== watchdog.log tail ==="
$watchdogLog = "$root\exp\628_ablation\destructive_logs\watchdog.log"
if (Test-Path $watchdogLog) {
    Get-Content $watchdogLog -Tail 20
}

Write-Host "`n=== batch_log.txt tail ==="
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 10
}

Write-Host "`n=== Current python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,@{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}

Write-Host "`n=== nvidia-smi ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

Write-Host "`n=== Progress check ==="
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0
$pending = 0
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) { $done++ } else { $pending++ }
}
Write-Host "Done: $done / $($allConfigs.Count) | Pending: $pending"
