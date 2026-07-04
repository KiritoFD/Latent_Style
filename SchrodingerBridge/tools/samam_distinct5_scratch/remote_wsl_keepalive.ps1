# WSL keep-alive task: runs `wsl sleep infinity` to prevent VM shutdown
# Create as scheduled task that runs at logon and stays alive

$taskName = "WSL_KeepAlive"
$wslExe = "C:\Windows\System32\wsl.exe"
$args = "-d Ubuntu -- exec sleep infinity"

# Remove existing task
schtasks /Delete /TN $taskName /F 2>$null

# Create task that runs immediately and keeps WSL alive
# Use SYSTEM account so it survives user logoff (but SYSTEM may not have WSL access)
# Instead use current user with "run whether user logged on or not"
$action = New-ScheduledTaskAction -Execute $wslExe -Argument $args
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::MaxValue) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force

# Run it immediately
Start-ScheduledTask -TaskName $taskName

Write-Host "Task $taskName created and started"
Start-Sleep -Seconds 5
schtasks /Query /TN $taskName /V /FO LIST | Select-String "Status|Last Run|Next"
