# Start a hidden, detached WSL holder process that survives SSH disconnect
# First, delete the scheduled task (it's not working properly)
schtasks /Delete /TN WSL_Holder /F 2>$null | Out-Null

# Check if a wsl holder is already running
$existing = Get-Process -Name wsl -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -lt (Get-Date).AddMinutes(-1) }
if ($existing) {
    Write-Host "WSL holder already running (PID $($existing.Id -join ','))"
} else {
    # Start hidden WSL holder process
    $proc = Start-Process -FilePath "wsl.exe" `
        -ArgumentList '-d','Ubuntu-22.04','-e','bash','-c','while true; do sleep 3600; done' `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Started WSL holder PID=$($proc.Id)"

    # Wait for it to stabilize
    Start-Sleep 3

    # Verify it's still running
    $check = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($check) {
        Write-Host "WSL holder confirmed alive (PID=$($proc.Id))"
    } else {
        Write-Host "ERROR: WSL holder died immediately!"
    }
}

# Show current wsl processes
Write-Host ""
Write-Host "=== All WSL processes ==="
Get-Process -Name wsl -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, StartTime -AutoSize

# Show WSL state
Write-Host ""
Write-Host "=== WSL State ==="
&wsl -l -v
