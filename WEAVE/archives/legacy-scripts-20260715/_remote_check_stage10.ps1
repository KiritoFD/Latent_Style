# Check stage10 training status
Write-Host "=== Python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU -AutoSize

Write-Host "`n=== Log file ==="
$logFile = "C:/Users/Administrator/logs/stage10_ll_partial_train.out"
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 30
} else {
    Write-Host "log file not yet created: $logFile"
}

Write-Host "`n=== Runner script ==="
$runner = "C:/Users/Administrator/scripts/_stage10_runner.ps1"
if (Test-Path $runner) {
    Write-Host "runner exists"
} else {
    Write-Host "runner MISSING"
}

Write-Host "`n=== Process 2224 ==="
Get-Process -Id 2224 -ErrorAction SilentlyContinue | Format-Table Id,ProcessName,StartTime -AutoSize
