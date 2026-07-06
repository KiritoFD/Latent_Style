# Launch baselines evaluation as a persistent scheduled task on remote
# This decouples it from the SSH session lifetime
$script = @'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$ps1 = "$repo\scripts\_eval_baselines_wikiarts15.ps1"
$taskName = "baseline_wikiarts15"

# Remove existing task if any
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create a one-time task that runs immediately under SYSTEM account
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$ps1`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

Write-Output "Task '$taskName' registered. Starting..."
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Output ("Task State: {0}" -f $task.State)
Write-Output ("Last Run Time: {0}" -f $info.LastRunTime)
Write-Output ("Last Task Result: {0}" -f $info.LastTaskResult)

# Verify the script file exists
Write-Output ""
Write-Output "=== script exists check ==="
if (Test-Path $ps1) {
    Write-Output "OK: $ps1 exists"
    Write-Output ("Size: {0} bytes" -f (Get-Item $ps1).Length)
} else {
    Write-Output "MISSING: $ps1"
}

# Also verify the python script exists
$py = "$repo\scripts\gen_trainfree_wikiarts15.py"
if (Test-Path $py) {
    Write-Output "OK: $py exists"
    Write-Output ("Size: {0} bytes" -f (Get-Item $py).Length)
} else {
    Write-Output "MISSING: $py"
}
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
