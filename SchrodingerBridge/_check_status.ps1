$errPath = "I:\Github\Latent_Style\SchrodingerBridge\train_stage1_err.txt"
Write-Output "=== ERR (last 60 lines) ==="
if (Test-Path $errPath) {
    Get-Content $errPath -Tail 60
}
Write-Output ""
Write-Output "=== Process status ==="
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    $py | Format-Table Id, CPU, WorkingSet64, StartTime
} else {
    Write-Output "No python process running"
}
Write-Output ""
Write-Output "=== GPU status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
Write-Output ""
Write-Output "=== Checkpoint dir ==="
$ckptDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2"
if (Test-Path $ckptDir) {
    Get-ChildItem $ckptDir | Format-Table Name, Length, LastWriteTime
} else {
    Write-Output "(checkpoint dir not found)"
}
