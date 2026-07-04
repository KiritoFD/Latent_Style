Set-Location I:\Github\Latent_Style\SchrodingerBridge
$env:PYTHONPATH = ""
$action = New-ScheduledTaskAction -Execute "C:\Program Files\Python312\python.exe" -Argument "run.py --config configs/p4_n11_n16_gate03_whh25.json" -WorkingDirectory "I:\Github\Latent_Style\SchrodingerBridge"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "n11_train" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "n11_train"
Write-Output "SCHTASK_STARTED"
