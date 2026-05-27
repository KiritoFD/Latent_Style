$ErrorActionPreference = "Stop"

$root = "I:\Github\Latent_Style\SchrodingerBridge"
$taskName = "LANCET_transport_moment_full"
$cmdPath = Join-Path $root "run_ema_transport_moment_full_remote.cmd"

Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root "exp\vae_backend\ema_transport_moment") | Out-Null

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument ("/c `"$cmdPath`"")
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(10)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
