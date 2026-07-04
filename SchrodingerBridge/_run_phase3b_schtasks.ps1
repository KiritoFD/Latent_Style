Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$wrapper = "I:\Github\Latent_Style\SchrodingerBridge\_run_phase3b_capture.ps1"
$argString = "-ExecutionPolicy Bypass -File `"$wrapper`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argString -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$existingTask = Get-ScheduledTask -TaskName "phase3b_abl" -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName "phase3b_abl" -Confirm:$false
}
Register-ScheduledTask -TaskName "phase3b_abl" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "phase3b_abl"
Write-Output "SCHTASK_STARTED: task=phase3b_abl"
