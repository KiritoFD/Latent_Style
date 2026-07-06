# Check X01 full_eval failure details
$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_X01_euler_train.log.err"
$EVAL_DIR = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512\X01_euler\full_eval"

Write-Host "=== X01 train log.err (last 60 lines) ==="
if (Test-Path $LOG) {
    Get-Content $LOG -Tail 60
} else {
    Write-Host "No .err file"
}

Write-Host ""
Write-Host "=== X01 full_eval directory structure ==="
if (Test-Path $EVAL_DIR) {
    Get-ChildItem -Path $EVAL_DIR -Recurse | Select-Object FullName, Length | Format-Table -AutoSize
} else {
    Write-Host "No full_eval dir"
}

Write-Host ""
Write-Host "=== X01 checkpoint dir ==="
$CKPT_DIR = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512\X01_euler"
if (Test-Path $CKPT_DIR) {
    Get-ChildItem -Path $CKPT_DIR | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
}
