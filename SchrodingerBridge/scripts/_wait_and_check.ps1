# Wait and check X01 status (training + eval result)
param([int]$WaitSec = 150)

Write-Host "Waiting $WaitSec seconds..."
Start-Sleep -Seconds $WaitSec

Write-Host ""
Write-Host "=== Batch log (last 20 lines) ==="
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_batch.log"
Get-Content $LOG -Tail 20

Write-Host ""
Write-Host "=== X01 status ==="
$EXP_ROOT = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512"
$X01_EVAL = "$EXP_ROOT\X01_euler\full_eval\epoch_0005\summary.json"
$X01_CKPT = "$EXP_ROOT\X01_euler\epoch_0005.pt"
Write-Host "Checkpoint exists: $(Test-Path $X01_CKPT)"
Write-Host "Eval summary exists: $(Test-Path $X01_EVAL)"
if (Test-Path $X01_EVAL) {
    Write-Host "=== X01 eval summary ==="
    Get-Content $X01_EVAL | ConvertFrom-Json | ConvertTo-Json -Depth 5
}

Write-Host ""
Write-Host "=== Latest err log (last 30 lines) ==="
$ERR_FILES = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_*_train.log.err" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($ERR_FILES) {
    Write-Host "File: $($ERR_FILES.Name) (modified $($ERR_FILES.LastWriteTime))"
    Get-Content $ERR_FILES.FullName -Tail 30
}

Write-Host ""
Write-Host "=== Python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet, StartTime | Format-Table
