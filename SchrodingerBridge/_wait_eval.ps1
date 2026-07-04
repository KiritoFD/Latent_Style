Start-Sleep -Seconds 120
$errPath = "I:\Github\Latent_Style\SchrodingerBridge\eval_stage1_err.txt"
$logPath = "I:\Github\Latent_Style\SchrodingerBridge\eval_stage1_log.txt"
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
Write-Output "=== Eval output dir ==="
$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010"
if (Test-Path $evalDir) {
    Get-ChildItem $evalDir -File | Format-Table Name, Length, LastWriteTime
} else {
    Write-Output "(eval dir not found yet)"
}
