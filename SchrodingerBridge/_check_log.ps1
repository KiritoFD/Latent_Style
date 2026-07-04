$logPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_log.txt"
$errPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"
Write-Output "=== LOG (last 40 lines) ==="
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 40
} else {
    Write-Output "(log file not found)"
}
Write-Output ""
Write-Output "=== ERR (last 40 lines) ==="
if (Test-Path $errPath) {
    Get-Content $errPath -Tail 40
} else {
    Write-Output "(err file not found)"
}
Write-Output ""
Write-Output "=== Process status ==="
tasklist | findstr python.exe
