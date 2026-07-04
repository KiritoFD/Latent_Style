# Check which 628 destructive experiments are done vs pending
$ErrorActionPreference = 'Continue'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"

$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
Write-Host "Total configs: $($allConfigs.Count)"

$done = @()
$pending = @()
foreach ($cfg in $allConfigs) {
    $name = $cfg.BaseName
    $ep10 = Join-Path $expDir "$name\epoch_0010.pt"
    if (Test-Path $ep10) {
        $done += $name
    } else {
        $pending += $name
    }
}

Write-Host "Done: $($done.Count)"
Write-Host "Pending: $($pending.Count)"
Write-Host "`n=== Pending list (first 30) ==="
$pending | Select-Object -First 30 | ForEach-Object { Write-Host "  $_" }
if ($pending.Count -gt 30) { Write-Host "  ... and $($pending.Count - 30) more" }

Write-Host "`n=== GPU process check ==="
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pyProcs) {
    foreach ($p in $pyProcs) {
        Write-Host "PID=$($p.Id) CPU=$($p.CPU) WS=$([math]::Round($p.WorkingSet64/1MB,1))MB StartTime=$($p.StartTime)"
    }
} else {
    Write-Host "No python process running"
}

Write-Host "`n=== Batch log tail ==="
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 5
}
