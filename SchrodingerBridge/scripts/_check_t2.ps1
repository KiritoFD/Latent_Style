# Check T2 FASG training status
$logFile = "C:\Users\Administrator\logs\t2_fasg_5ep_train_eval.out"
if (Test-Path $logFile) {
    Write-Output "LOG_EXISTS"
    Get-Content $logFile -Tail 30
} else {
    Write-Output "LOG_NOT_FOUND"
    $py = Get-Process python -ErrorAction SilentlyContinue
    if ($py) {
        Write-Output "PYTHON_RUNNING pid=$($py.Id)"
    } else {
        Write-Output "PYTHON_NOT_RUNNING"
    }
}
