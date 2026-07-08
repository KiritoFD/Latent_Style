# Wait for StyleAligned to finish, then launch Z-STAR
# This script monitors python.exe and starts Z-STAR when no python.exe is running

Write-Output "[$(Get-Date)] Z-STAR launcher started, waiting for StyleAligned to finish..."

while ($true) {
    $pyProcs = Get-Process -Name python -ErrorAction SilentlyContinue
    if (-not $pyProcs) {
        Write-Output "[$(Get-Date)] No python.exe running. Launching Z-STAR..."
        break
    }
    foreach ($p in $pyProcs) {
        $cmd = (gwmi Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
        Write-Output "[$(Get-Date)] Python PID=$($p.Id) running: $cmd"
    }
    Write-Output "[$(Get-Date)] Waiting 60s..."
    Start-Sleep -Seconds 60
}

# Small delay to let GPU memory release
Start-Sleep -Seconds 15

# Launch Z-STAR
Write-Output "[$(Get-Date)] Starting Z-STAR inference..."
& "C:\Program Files\Python312\python.exe" "C:\Users\Administrator\_run_zstar_remote.py" --fp16 2>&1 | Out-File -FilePath "C:\Users\Administrator\logs\zstar_run.log" -Encoding utf8

Write-Output "[$(Get-Date)] Z-STAR finished."
