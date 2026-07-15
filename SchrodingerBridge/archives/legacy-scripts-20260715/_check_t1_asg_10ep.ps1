# Check ASG 10ep training status
$logFile = "C:\Users\Administrator\logs\t1_asg_10ep_train_eval.out"
if (Test-Path $logFile) {
    Write-Output "LOG_EXISTS"
    Get-Content $logFile -Tail 30
} else {
    Write-Output "LOG_NOT_FOUND"
    # Check if python process running
    $py = Get-Process python -ErrorAction SilentlyContinue
    if ($py) {
        Write-Output "PYTHON_RUNNING pid=$($py.Id)"
    } else {
        Write-Output "PYTHON_NOT_RUNNING"
    }
    # Check exp dir
    $expDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_10ep"
    if (Test-Path $expDir) {
        Write-Output "EXP_DIR_EXISTS"
        Get-ChildItem $expDir -Recurse | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
    } else {
        Write-Output "EXP_DIR_NOT_EXISTS"
    }
}
