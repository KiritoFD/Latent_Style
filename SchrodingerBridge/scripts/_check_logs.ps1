# Check all logs
Write-Output "=== zstar_launcher.log ==="
if (Test-Path "C:\Users\Administrator\logs\zstar_launcher.log") {
    Get-Content "C:\Users\Administrator\logs\zstar_launcher.log" -Tail 20
} else { Write-Output "NOT FOUND" }

Write-Output "`n=== zstar_run.log ==="
if (Test-Path "C:\Users\Administrator\logs\zstar_run.log") {
    Get-Content "C:\Users\Administrator\logs\zstar_run.log" -Tail 30
} else { Write-Output "NOT FOUND" }

Write-Output "`n=== sa_run5.log (last lines) ==="
if (Test-Path "C:\Users\Administrator\logs\sa_run5.log") {
    Get-Content "C:\Users\Administrator\logs\sa_run5.log" -Tail 20
} else { Write-Output "NOT FOUND" }

Write-Output "`n=== All logs directory ==="
Get-ChildItem "C:\Users\Administrator\logs" | Select-Object Name,Length,LastWriteTime
