# Check if zstar_launcher is running
$procs = Get-Process -Name powershell -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $PID }
foreach ($p in $procs) {
    $cmd = (gwmi Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
    if ($cmd -like "*zstar*") {
        Write-Output "Z-STAR launcher running: PID=$($p.Id) CMD=$cmd"
    }
}
# Also check the launcher log
if (Test-Path "C:\Users\Administrator\logs\zstar_launcher.log") {
    Get-Content "C:\Users\Administrator\logs\zstar_launcher.log" -Tail 5
} else {
    Write-Output "No zstar_launcher.log found (launcher may write to console only)"
}
