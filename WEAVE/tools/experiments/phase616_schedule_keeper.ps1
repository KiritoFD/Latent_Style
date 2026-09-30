param(
    [string]$TaskName = 'phase616_h0_resume_keeper',
    [string]$RunScript = '/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/phase616_resume_keeper.sh'
)

$startAt = (Get-Date).AddMinutes(1)
$action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument ("bash " + $RunScript)
$trigger = New-ScheduledTaskTrigger -Once -At $startAt
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -User 'Administrator' | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Output ("TASK " + $TaskName)
Write-Output ("START_AT " + $startAt.ToString('yyyy-MM-dd HH:mm:ss'))
