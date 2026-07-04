param([string]$TaskName = "phase3_abl")
Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$wrapper = "I:\Github\Latent_Style\SchrodingerBridge\_run_phase3_capture.ps1"
$argString = "-ExecutionPolicy Bypass -File `"$wrapper`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argString -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "SCHTASK_STARTED: task=$TaskName (phase3 ablations)"
