# Setup a recurring watchdog schtask that runs every 5 minutes to keep batch runner alive
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$watchdogScript = "$root\_628_watchdog.ps1"

# Verify watchdog script exists
if (-not (Test-Path $watchdogScript)) {
    Write-Host "ERROR: watchdog script not found: $watchdogScript"
    exit 1
}

# Delete existing watchdog tasks
schtasks /Delete /TN 'sb_628_watchdog' /F 2>$null | Out-Null
schtasks /Delete /TN 'sb_628_watchdog_min' /F 2>$null | Out-Null

# Create watchdog schtask that runs every 5 minutes
# Use /MO 5 with /SC MINUTE for recurring every 5 minutes
$cmdLine = "powershell.exe -ExecutionPolicy Bypass -NoProfile -File `"$watchdogScript`""
Write-Host "Creating watchdog schtask: sb_628_watchdog (every 5 min)"
Write-Host "  cmd: $cmdLine"

$createResult = schtasks /Create /TN 'sb_628_watchdog' /TR $cmdLine /SC MINUTE /MO 5 /RU $env:USERNAME /IT /F 2>&1
Write-Host "Create result: $createResult"

# Verify task was created
Write-Host "`n=== Verify task ==="
schtasks /Query /TN 'sb_628_watchdog' /V /FO LIST 2>&1 | Select-String -Pattern 'TaskName|Status|Schedule Type|Repeat: Every|Next Run Time|Last Run Time|Last Result'

# Run watchdog immediately to check current state
Write-Host "`n=== Running watchdog immediately ==="
schtasks /Run /TN 'sb_628_watchdog'
Start-Sleep -Seconds 10

# Check watchdog log
$watchdogLog = "$root\exp\628_ablation\destructive_logs\watchdog.log"
if (Test-Path $watchdogLog) {
    Write-Host "`n=== watchdog.log ==="
    Get-Content $watchdogLog -Tail 10
} else {
    Write-Host "No watchdog.log yet"
}

# Also check current process state
Write-Host "`n=== Current python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,@{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}

Write-Host "`n=== nvidia-smi ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

Write-Host "`n=== batch_log.txt tail ==="
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 5
}
