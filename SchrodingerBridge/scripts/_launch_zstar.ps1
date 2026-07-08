# Launch Z-STAR via schtasks (proven method from SaMam runs)
# Step 1: Delete existing task if any
schtasks /delete /tn "zstar_run" /f 2>$null

# Step 2: Create scheduled task that runs immediately
$action = New-ScheduledTaskAction -Execute "C:\Program Files\Python312\python.exe" -Argument "C:\Users\Administrator\_run_zstar_remote.py --fp16" -WorkingDirectory "C:\Users\Administrator"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
Register-ScheduledTask -TaskName "zstar_run" -Action $action -Trigger $trigger -Settings $settings -User "Administrator" -RunLevel Highest -Force

# Step 3: Run it now
schtasks /run /tn "zstar_run"

Write-Output "Z-STAR task created and triggered"
