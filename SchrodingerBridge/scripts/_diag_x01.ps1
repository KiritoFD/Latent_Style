# Diagnose X01 training failure
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_X01_euler_train.log"
Write-Host "=== X01 train log (first 80 lines) ==="
if (Test-Path $LOG) {
    Get-Content $LOG -TotalCount 80
} else {
    Write-Host "Log not found: $LOG"
}
