$ErrorActionPreference = 'SilentlyContinue'
Write-Host "===== ALL python / cmd processes (with parent tree) ====="
Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'cmd.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
    Format-List

Write-Host ""
Write-Host "===== Watchdog / scheduled task check ====="
Get-Process -Name "powershell","pwsh" -ErrorAction SilentlyContinue |
    Select-Object Id, StartTime, Path | Format-Table -AutoSize

Write-Host ""
Write-Host "===== Schtasks containing 628/destructive/batch ====="
schtasks /Query /FO CSV /V 2>$null | Select-String -Pattern "628|destructive|p7_runner|p8d" -SimpleMatch

Write-Host ""
Write-Host "===== Any .bat still running ====="
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -match "628|p7|p8d|destructive" } |
    Select-Object ProcessId, ParentProcessId, CommandLine | Format-List
