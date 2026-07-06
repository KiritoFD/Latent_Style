# Setup schtasks watchdog for abl512 training (runs every 5 minutes)
# Uses SYSTEM account so training survives SSH disconnects

$WATCHDOG = "I:\Github\Latent_Style\SchrodingerBridge\scripts\_watchdog_abl512.ps1"
$TASK_NAME = "abl512_watchdog"

Write-Host "=== Setting up schtasks watchdog ==="
Write-Host "Task name: $TASK_NAME"
Write-Host "Watchdog script: $WATCHDOG"
Write-Host "Schedule: every 5 minutes"
Write-Host "User: SYSTEM (survives SSH disconnect)"
Write-Host ""

# Delete existing task if any
schtasks /Delete /TN $TASK_NAME /F 2>$null
Write-Host "Deleted existing task (if any)"

# Create task using command line (simpler than XML)
# /RU SYSTEM = run as SYSTEM (no password needed, survives logout)
# /SC MINUTE /MO 5 = every 5 minutes
$cmd = "powershell.exe -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$WATCHDOG`""
Write-Host "Command: $cmd"
Write-Host ""

$result = schtasks /Create /TN $TASK_NAME /TR $cmd /SC MINUTE /MO 5 /RU SYSTEM /F
Write-Host "Create result: $result"

# Verify
Write-Host ""
Write-Host "=== Task query ==="
schtasks /Query /TN $TASK_NAME /V /FO LIST 2>&1 | Select-Object -First 30

# Run immediately to start training now
Write-Host ""
Write-Host "=== Running task immediately ==="
schtasks /Run /TN $TASK_NAME
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "=== Watchdog log after 3s ==="
$watchdogLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_watchdog.log"
if (Test-Path $watchdogLog) {
    Get-Content $watchdogLog -Tail 10
} else {
    Write-Host "Watchdog log not yet created"
}
