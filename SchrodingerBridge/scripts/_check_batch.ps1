# Check batch training status
$BATCH_LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_batch.log"
$EXP_ROOT = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512"

Write-Host "=== Batch log (last 30 lines) ==="
if (Test-Path $BATCH_LOG) {
    Get-Content $BATCH_LOG -Tail 30
} else {
    Write-Host "Batch log not found"
}

Write-Host ""
Write-Host "=== Running python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, CPU, WorkingSet, StartTime -AutoSize

Write-Host ""
Write-Host "=== Completed experiments (with summary.json) ==="
if (Test-Path $EXP_ROOT) {
    $completed = Get-ChildItem -Path $EXP_ROOT -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue
    if ($completed) {
        $completed | ForEach-Object {
            $expName = $_.FullName -replace '.*\\abl512\\', '' -replace '\\.*', ''
            $epoch = $_.FullName -replace '.*\\epoch_', '' -replace '\\.*', ''
            Write-Host "  $expName (epoch $epoch): $($_.FullName)"
        }
    } else {
        Write-Host "  None yet"
    }
}

Write-Host ""
Write-Host "=== Exp directories created ==="
if (Test-Path $EXP_ROOT) {
    Get-ChildItem -Path $EXP_ROOT -Directory | Select-Object Name, LastWriteTime | Format-Table -AutoSize
}
