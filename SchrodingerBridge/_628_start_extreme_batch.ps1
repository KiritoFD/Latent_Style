# Start the 628 extreme batch runner via schtasks (decoupled from SSH session)
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$batPath = "$root\_628_batch_runner.bat"
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

# Clean old log files
foreach ($f in @($stdoutLog, $stderrLog)) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

# Write the .bat launcher (same as before, runs 628_run_destructive_batch.py)
$batContent = @"
@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set RUNNER=I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%RUNNER%" > "%STDOUT%" 2> "%STDERR%"
"@
[System.IO.File]::WriteAllText($batPath, $batContent, [System.Text.Encoding]::Default)

# Delete existing schtasks
schtasks /Delete /TN 'sb_628_batch_runner' /F 2>$null | Out-Null
schtasks /Delete /TN 'sb_628_watchdog' /F 2>$null | Out-Null

# Create batch runner schtask
$taskName = 'sb_628_batch_runner'
Write-Host "Creating schtask: $taskName"
schtasks /Create /TN $taskName /TR $batPath /SC ONCE /ST 23:59 /RU $env:USERNAME /IT /F 2>&1 | Out-Null

Write-Host "Running schtask immediately..."
schtasks /Run /TN $taskName

Start-Sleep -Seconds 10

# Find the python process
$py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
if ($py) {
    Write-Host "SUCCESS: python PID=$($py.Id) StartTime=$($py.StartTime)"
    "$($py.Id)" | Out-File -FilePath $pidFile -Encoding ascii
} else {
    Write-Host "WARNING: no python process detected after 10s"
    if (Test-Path $stderrLog) {
        Write-Host "--- stderr ---"
        Get-Content $stderrLog -Tail 20
    }
}

# Also create watchdog schtask
$watchdogScript = "$root\_628_watchdog.ps1"
$watchdogCmd = "powershell.exe -ExecutionPolicy Bypass -NoProfile -File `"$watchdogScript`""
schtasks /Create /TN 'sb_628_watchdog' /TR $watchdogCmd /SC MINUTE /MO 5 /RU $env:USERNAME /IT /F 2>&1 | Out-Null
schtasks /Run /TN 'sb_628_watchdog' 2>$null | Out-Null
Write-Host "Watchdog started (every 5 min)"

# Count total configs
$cfgDir = "$root\configs\ablations\628_destructive"
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$expDir = "$root\exp\628_ablation\destructive"
$done = 0
$pending = 0
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) { $done++ } else { $pending++ }
}
Write-Host "`nTotal configs: $($allConfigs.Count) | Done: $done | Pending: $pending"
Write-Host "Estimated time: $([math]::Round($pending * 345 / 60, 1)) min ($([math]::Round($pending * 345 / 3600, 1)) h)"
