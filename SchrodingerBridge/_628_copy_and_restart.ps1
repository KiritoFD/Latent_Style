$root = 'I:\Github\Latent_Style\SchrodingerBridge'

Write-Host "=== Step 1: Copy updated batch runner ==="
Copy-Item /tmp/628_run_destructive_batch.py "$root\628_run_destructive_batch.py" -Force
$info = Get-Item "$root\628_run_destructive_batch.py"
Write-Host "OK: $($info.Length) bytes"

Write-Host ""
Write-Host "=== Step 2: Stop schtasks and kill python ==="
schtasks /End /TN 'sb_628_batch_runner' 2>&1 | Out-Null
schtasks /End /TN 'sb_628_watchdog' 2>&1 | Out-Null
schtasks /Delete /TN 'sb_628_batch_runner' /F 2>&1 | Out-Null
schtasks /Delete /TN 'sb_628_watchdog' /F 2>&1 | Out-Null
Start-Sleep -Seconds 2

$pyProcs = Get-Process python -ErrorAction SilentlyContinue
foreach ($p in $pyProcs) {
    Write-Host "  Killing PID=$($p.Id) StartTime=$($p.StartTime)"
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}
$cmdProcs = Get-WmiObject Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -match '_628_batch_runner' }
foreach ($c in $cmdProcs) {
    Write-Host "  Killing cmd.exe PID=$($c.ProcessId)"
    Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3

$remaining = Get-Process python -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "WARNING: $($remaining.Count) python processes still alive"
} else {
    Write-Host "OK: no python processes remaining"
}

Write-Host ""
Write-Host "=== Step 3: Restart batch (X configs first) ==="
$batPath = "$root\_628_batch_runner.bat"
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

foreach ($f in @($stdoutLog, $stderrLog)) {
    if (Test-Path $f) {
        try { Remove-Item $f -Force -ErrorAction Stop } catch {}
    }
}

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

schtasks /Create /TN 'sb_628_batch_runner' /TR $batPath /SC ONCE /ST 23:59 /RU $env:USERNAME /IT /F 2>&1 | Out-Null
schtasks /Run /TN 'sb_628_batch_runner'
Start-Sleep -Seconds 10

$py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
if ($py) {
    Write-Host "SUCCESS: python PID=$($py.Id) StartTime=$($py.StartTime)"
    "$($py.Id)" | Out-File -FilePath $pidFile -Encoding ascii
} else {
    Write-Host "WARNING: no python process detected"
    if (Test-Path $stderrLog) {
        Get-Content $stderrLog -Tail 20
    }
}

$watchdogScript = "$root\_628_watchdog.ps1"
$watchdogCmd = "powershell.exe -ExecutionPolicy Bypass -NoProfile -File `"$watchdogScript`""
schtasks /Create /TN 'sb_628_watchdog' /TR $watchdogCmd /SC MINUTE /MO 5 /RU $env:USERNAME /IT /F 2>&1 | Out-Null
schtasks /Run /TN 'sb_628_watchdog' 2>$null | Out-Null
Write-Host "Watchdog started"

Write-Host ""
Write-Host "=== Step 4: Verify X configs run first ==="
Start-Sleep -Seconds 5
if (Test-Path $stdoutLog) {
    Write-Host "First 15 lines of stdout:"
    Get-Content $stdoutLog -Head 15
}
