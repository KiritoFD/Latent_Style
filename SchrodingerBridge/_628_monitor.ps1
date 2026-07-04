$root = 'I:\Github\Latent_Style\SchrodingerBridge'

Write-Host "=== Batch Log (last 20 lines) ==="
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 20
}

Write-Host ""
Write-Host "=== Python Process ==="
$py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
if ($py) {
    Write-Host "PID=$($py.Id) StartTime=$($py.StartTime) CPU=$($py.CPU) WS=$([math]::Round($py.WorkingSet64/1MB,1))MB"
} else {
    Write-Host "NO python process!"
}

Write-Host ""
Write-Host "=== Current X experiment log (last 30 lines) ==="
$xLogs = Get-ChildItem "$root\exp\628_ablation\destructive_logs" -Filter 'X*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($xLogs) {
    $latestXLog = $xLogs[0].FullName
    Write-Host "Latest X log: $($xLogs[0].Name)"
    Get-Content $latestXLog -Tail 30
} else {
    Write-Host "No X logs yet"
}

Write-Host ""
Write-Host "=== 628-DEBUG lines (verify loss is REAL) ==="
Get-ChildItem "$root\exp\628_ablation\destructive_logs" -Filter 'X*.log' -ErrorAction SilentlyContinue | ForEach-Object {
    $matches = Select-String -Path $_.FullName -Pattern '628-DEBUG' -ErrorAction SilentlyContinue
    if ($matches) {
        foreach ($m in $matches) {
            Write-Host "[$($_.Name)] $($m.Line)"
        }
    }
}

Write-Host ""
Write-Host "=== Progress Summary ==="
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0; $pending = 0
$xTotal = 0; $xDone = 0
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    $isDone = Test-Path $ep10
    if ($isDone) { $done++ } else { $pending++ }
    if ($cfg.BaseName -like 'X*') {
        $xTotal++
        if ($isDone) { $xDone++ }
    }
}
Write-Host "Total: $($allConfigs.Count) | Done: $done | Pending: $pending"
Write-Host "X configs: $xDone / $xTotal done"

Write-Host ""
Write-Host "=== X configs status ==="
$xConfigs = Get-ChildItem $cfgDir -Filter 'X*.json' | Sort-Object Name
foreach ($cfg in $xConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    $ep8 = Join-Path $expDir "$($cfg.BaseName)\epoch_0008.pt"
    $ep9 = Join-Path $expDir "$($cfg.BaseName)\epoch_0009.pt"
    $status = "PENDING"
    if (Test-Path $ep10) { $status = "DONE" }
    elseif (Test-Path $ep9) { $status = "EP9" }
    elseif (Test-Path $ep8) { $status = "EP8" }
    elseif (Test-Path (Join-Path $expDir "$($cfg.BaseName)")) { $status = "STARTED" }
    Write-Host "  $($cfg.BaseName): $status"
}
