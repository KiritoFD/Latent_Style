Write-Host "Waiting 180 seconds for X10 to finish and X11 to start..."
Start-Sleep -Seconds 180

$root = 'I:\Github\Latent_Style\SchrodingerBridge'

Write-Host ""
Write-Host "=== Batch Log (last 10 lines) ==="
Get-Content "$root\exp\628_ablation\destructive_logs\batch_log.txt" -Tail 10

Write-Host ""
Write-Host "=== 628-ALL-DEBUG lines in X logs ==="
Get-ChildItem "$root\exp\628_ablation\destructive_logs" -Filter 'X*.log' -ErrorAction SilentlyContinue | ForEach-Object {
    $matches = Select-String -Path $_.FullName -Pattern '628-ALL-DEBUG' -ErrorAction SilentlyContinue
    if ($matches) {
        foreach ($m in $matches) {
            Write-Host "[$($_.Name)] $($m.Line)"
        }
    }
}

Write-Host ""
Write-Host "=== Latest X log (last 15 lines) ==="
$xLogs = Get-ChildItem "$root\exp\628_ablation\destructive_logs" -Filter 'X*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($xLogs) {
    Write-Host "Latest: $($xLogs[0].Name)"
    Get-Content $xLogs[0].FullName -Tail 15
}

Write-Host ""
Write-Host "=== X configs done count ==="
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"
$xConfigs = Get-ChildItem $cfgDir -Filter 'X*.json'
$xDone = 0
foreach ($cfg in $xConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) { $xDone++ }
}
Write-Host "X done: $xDone / $($xConfigs.Count)"
