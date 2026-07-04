$ErrorActionPreference = "Continue"
# Start pixel256 training only (latent256 has nan issue to debug)
$pixelAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File C:\Users\Administrator\scripts\run_pixel256.ps1"
$pixelTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$pixelSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
Register-ScheduledTask -TaskName "pixel256_train" -Action $pixelAction -Trigger $pixelTrigger -Settings $pixelSettings -Force | Out-Null
Write-Output "Created pixel256_train task"
Start-ScheduledTask -TaskName "pixel256_train"
Write-Output "Started pixel256_train"
