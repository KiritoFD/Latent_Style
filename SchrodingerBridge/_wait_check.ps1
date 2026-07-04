Start-Sleep -Seconds 90
$errPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"
Write-Output "=== ERR (last 15 lines) ==="
if (Test-Path $errPath) {
    Get-Content $errPath -Tail 15
} else {
    Write-Output "(no err file)"
}
Write-Output ""
Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
