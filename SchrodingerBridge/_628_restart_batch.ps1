# Kill all running python processes and restart batch cleanly
$root = 'I:\Github\Latent_Style\SchrodingerBridge'

Write-Host "=== Step 1: Stop schtasks ==="
schtasks /End /TN 'sb_628_batch_runner' 2>&1 | Out-Null
schtasks /End /TN 'sb_628_watchdog' 2>&1 | Out-Null
schtasks /Delete /TN 'sb_628_batch_runner' /F 2>&1 | Out-Null
schtasks /Delete /TN 'sb_628_watchdog' /F 2>&1 | Out-Null
Start-Sleep -Seconds 2

Write-Host "=== Step 2: Kill all python.exe processes ==="
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
foreach ($p in $pyProcs) {
    Write-Host "  Killing PID=$($p.Id) StartTime=$($p.StartTime)"
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
# Also kill any cmd.exe that's running the batch runner .bat
$cmdProcs = Get-WmiObject Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -match '_628_batch_runner' }
foreach ($c in $cmdProcs) {
    Write-Host "  Killing cmd.exe PID=$($c.ProcessId)"
    Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

Write-Host "=== Step 3: Verify all killed ==="
$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "WARNING: $($remaining.Count) python processes still alive:"
    $remaining | Format-Table Id, StartTime, CPU
} else {
    Write-Host "OK: no python processes remaining"
}

Write-Host ""
Write-Host "=== Step 4: Count configs and done experiments ==="
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0
$pending = 0
$pendingNames = @()
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) {
        $done++
    } else {
        $pending++
        $pendingNames += $cfg.BaseName
    }
}
Write-Host "Total: $($allConfigs.Count) | Done: $done | Pending: $pending"
Write-Host ""
Write-Host "Pending experiments ($pending):"
$pendingNames | Sort-Object | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "=== Step 5: Restart batch via schtasks ==="
$batPath = "$root\_628_batch_runner.bat"
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

# Clean old log files
foreach ($f in @($stdoutLog, $stderrLog)) {
    if (Test-Path $f) {
        try { Remove-Item $f -Force -ErrorAction Stop } catch {}
    }
}

# Write the .bat launcher
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

# Create and run batch runner schtask
$taskName = 'sb_628_batch_runner'
schtasks /Create /TN $taskName /TR $batPath /SC ONCE /ST 23:59 /RU $env:USERNAME /IT /F 2>&1 | Out-Null
schtasks /Run /TN $taskName
Start-Sleep -Seconds 10

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

# Create and run watchdog
$watchdogScript = "$root\_628_watchdog.ps1"
$watchdogCmd = "powershell.exe -ExecutionPolicy Bypass -NoProfile -File `"$watchdogScript`""
schtasks /Create /TN 'sb_628_watchdog' /TR $watchdogCmd /SC MINUTE /MO 5 /RU $env:USERNAME /IT /F 2>&1 | Out-Null
schtasks /Run /TN 'sb_628_watchdog' 2>$null | Out-Null
Write-Host "Watchdog started (every 5 min)"

Write-Host ""
Write-Host "=== Step 6: Verify batch runner sees 159 configs ==="
Start-Sleep -Seconds 5
if (Test-Path $stdoutLog) {
    $firstLines = Get-Content $stdoutLog -Head 5
    foreach ($line in $firstLines) {
        Write-Host "  $line"
    }
}
