# Launcher - starts batch runner as hidden background process
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\Users\Administrator\_remote_batch_runner.ps1") -WindowStyle Hidden -PassThru
Write-Host "Started batch runner PID=$($proc.Id) at $(Get-Date)"
$proc.Id | Out-File "C:\Users\Administrator\_batch_runner_pid.txt"
