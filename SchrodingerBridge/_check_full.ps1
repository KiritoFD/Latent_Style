$errPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"
Write-Output "=== ERR (full content) ==="
if (Test-Path $errPath) {
    Get-Content $errPath
} else {
    Write-Output "(err file not found)"
}
Write-Output ""
Write-Output "=== LOG (full content) ==="
$logPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_log.txt"
if (Test-Path $logPath) {
    Get-Content $logPath
} else {
    Write-Output "(log file not found)"
}
Write-Output ""
Write-Output "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
