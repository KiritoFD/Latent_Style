Start-Sleep -Seconds 120
$errPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"
$logPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_log.txt"
Write-Output "=== ERR (last 30 lines) ==="
if (Test-Path $errPath) {
    Get-Content $errPath -Tail 30
}
Write-Output ""
Write-Output "=== LOG (last 30 lines) ==="
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 30
}
Write-Output ""
Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
Write-Output ""
Write-Output "=== Checkpoint dir ==="
$ckptDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2"
if (Test-Path $ckptDir) {
    Get-ChildItem $ckptDir -File | Format-Table Name, Length, LastWriteTime
}
