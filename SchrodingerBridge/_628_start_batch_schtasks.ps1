# Start batch runner via schtasks + dynamically created .bat launcher
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$batPath = "$root\_628_batch_runner.bat"
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

# Clean old log files
foreach ($f in @($stdoutLog, $stderrLog)) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

# Write the .bat launcher file with ASCII encoding (cmd.exe needs ASCII, not UTF-8 BOM)
$batContent = @"
@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set RUNNER=I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%RUNNER%" > "%STDOUT%" 2> "%STDERR%"
"@
# Use Default encoding (system ANSI, no BOM) - critical for cmd.exe
[System.IO.File]::WriteAllText($batPath, $batContent, [System.Text.Encoding]::Default)
Write-Host "Wrote .bat: $batPath"
Write-Host "Content:"
Get-Content $batPath

# Delete any existing schtask
$taskName = 'sb_628_batch_runner'
schtasks /Delete /TN $taskName /F 2>$null | Out-Null

# Create the schtask pointing to the .bat file
Write-Host "Creating schtask: $taskName"
$createResult = schtasks /Create /TN $taskName /TR $batPath /SC ONCE /ST 23:59 /RU $env:USERNAME /IT /F 2>&1
Write-Host "Create result: $createResult"

Write-Host "Running schtask immediately..."
$runResult = schtasks /Run /TN $taskName 2>&1
Write-Host "Run result: $runResult"

Start-Sleep -Seconds 8

# Find the python process
$py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
if ($py) {
    Write-Host "SUCCESS: python PID=$($py.Id) StartTime=$($py.StartTime)"
    "$($py.Id)" | Out-File -FilePath $pidFile -Encoding ascii
} else {
    Write-Host "WARNING: no python process detected after 8s"
    if (Test-Path $stderrLog) {
        Write-Host "--- stderr ---"
        Get-Content $stderrLog -Tail 30
    }
    if (Test-Path $stdoutLog) {
        Write-Host "--- stdout ---"
        Get-Content $stdoutLog -Tail 10
    }
    Write-Host "--- schtasks query ---"
    schtasks /Query /TN $taskName /V /FO LIST 2>&1 | Select-String -Pattern 'Last Result|Last Run Time|Status|Task To Run|Run As User'
}

Start-Sleep -Seconds 15
$py2 = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
if ($py2) {
    Write-Host "Confirmed: python PID=$($py2.Id) still running after 23s"
    if (Test-Path $stdoutLog) {
        Write-Host "--- stdout tail ---"
        Get-Content $stdoutLog -Tail 5
    }
} else {
    Write-Host "WARNING: python process exited within 23s"
    if (Test-Path $stderrLog) {
        Write-Host "--- stderr ---"
        Get-Content $stderrLog -Tail 30
    }
    if (Test-Path $stdoutLog) {
        Write-Host "--- stdout ---"
        Get-Content $stdoutLog -Tail 10
    }
}
