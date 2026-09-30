Write-Host "Killing all python.exe processes..."
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "Still running:"
    $remaining | ForEach-Object { Write-Host ("  PID=" + $_.Id) }
} else {
    Write-Host "All python processes killed."
}
