# Recreate schtasks with wrapper script for logging
schtasks /delete /tn "zstar_run" /f 2>$null

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\Users\Administrator\_run_zstar_wrapper.ps1" -WorkingDirectory "C:\Users\Administrator"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 24)
Register-ScheduledTask -TaskName "zstar_run" -Action $action -Trigger $trigger -Settings $settings -User "Administrator" -RunLevel Highest -Force

schtasks /run /tn "zstar_run"
Write-Output "Z-STAR wrapper task created and triggered"
