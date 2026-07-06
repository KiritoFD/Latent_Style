# Create and run schtasks for WikiArt-15 SD-Turbo + SaMam
$taskName = "wikiarts15_sdturbo_samam"
$scriptPath = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_sdturbo_samam_wikiarts15.ps1"

# Create the task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM" -Force | Out-Null

Write-Output "Task '$taskName' created."

# Start it now
Start-ScheduledTask -TaskName $taskName
Write-Output "Task '$taskName' started."
Start-Sleep -Seconds 3

# Check status
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Output "State: $($task.State)"
Write-Output "LastRunTime: $($info.LastRunTime)"
Write-Output "LastTaskResult: $($info.LastTaskResult)"
