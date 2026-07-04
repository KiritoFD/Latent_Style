param([string]$ConfigName, [string]$TaskName, [string]$ExpName)
if (-not $ExpName) { $ExpName = $ConfigName }
Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$wrapper = "I:\Github\Latent_Style\SchrodingerBridge\_run_train_capture_wrapper.ps1"
# Use PowerShell wrapper that calls Start-Process -RedirectStandardError
# This captures stderr to file while avoiding fortrl console detection
$argString = "-ExecutionPolicy Bypass -File `"$wrapper`" -ConfigName $ConfigName -ExpName $ExpName"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argString -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "SCHTASK_STARTED: task=$TaskName config=$ConfigName exp=$ExpName (capture_wrapper)"
