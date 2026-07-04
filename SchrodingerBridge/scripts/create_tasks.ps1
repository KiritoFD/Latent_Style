$ErrorActionPreference = "Continue"

# Create scheduled tasks for 256 training (survives SSH disconnect)

# latent256 task
$latentAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File C:\Users\Administrator\scripts\run_latent256.ps1"
$latentTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$latentSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
Register-ScheduledTask -TaskName "latent256_train" -Action $latentAction -Trigger $latentTrigger -Settings $latentSettings -Force | Out-Null
Write-Output "Created latent256_train task"

# pixel256 task
$pixelAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File C:\Users\Administrator\scripts\run_pixel256.ps1"
$pixelTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$pixelSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
Register-ScheduledTask -TaskName "pixel256_train" -Action $pixelAction -Trigger $pixelTrigger -Settings $pixelSettings -Force | Out-Null
Write-Output "Created pixel256_train task"

# Start latent256 immediately (pixel256 will start 5s later but we should run them sequentially to avoid VRAM contention)
# Actually, let's start only latent256 first, then pixel256 after latent256 finishes or we manually start it
Start-ScheduledTask -TaskName "latent256_train"
Write-Output "Started latent256_train"
