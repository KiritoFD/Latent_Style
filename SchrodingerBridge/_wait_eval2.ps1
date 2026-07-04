Start-Sleep -Seconds 60
$errPath = "I:\Github\Latent_Style\SchrodingerBridge\eval_stage1_err.txt"
$logPath = "I:\Github\Latent_Style\SchrodingerBridge\eval_stage1_log.txt"
$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010"

Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== LOG (last 40 lines) ==="
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 40
}
Write-Output ""
Write-Output "=== Eval output files ==="
if (Test-Path $evalDir) {
    Get-ChildItem $evalDir -File | Format-Table Name, Length, LastWriteTime
}
Write-Output ""
Write-Output "=== metrics.csv (last lines) ==="
$metricsPath = "$evalDir\metrics.csv"
if (Test-Path $metricsPath) {
    Get-Content $metricsPath -Tail 5
}
