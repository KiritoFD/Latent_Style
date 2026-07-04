# Watchdog: check if batch runner is alive; if not, restart it.
# This script is meant to be invoked by a recurring schtasks task every 5 minutes.
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$batPath = "$root\_628_batch_runner.bat"
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$stderrLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stderr.log"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"
$watchdogLog = "$root\exp\628_ablation\destructive_logs\watchdog.log"
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"

function Watchdog-Log {
    param([string]$msg)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] $msg"
    Add-Content -Path $watchdogLog -Value $line
    Write-Host $line
}

# Count pending experiments
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0
$pending = 0
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) { $done++ } else { $pending++ }
}
Watchdog-Log "Progress: $done/$($allConfigs.Count) done, $pending pending"

# If no pending, we're finished - exit
if ($pending -eq 0) {
    Watchdog-Log "All experiments complete. Watchdog exiting."
    exit 0
}

# Check if batch runner is alive
$batchAlive = $false
if (Test-Path $pidFile) {
    $bpid = (Get-Content $pidFile -Raw).Trim()
    $p = Get-Process -Id $bpid -ErrorAction SilentlyContinue
    if ($p -and -not $p.HasExited) {
        $batchAlive = $true
        Watchdog-Log "Batch runner PID=$bpid is alive (CPU=$($p.CPU) WS=$([math]::Round($p.WorkingSet64/1MB,1))MB)"
    }
}

# Check if a training subprocess is running (python with high memory usage)
$pyProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 500MB }
$trainingRunning = $false
if ($pyProcs) {
    $trainingRunning = $true
    foreach ($p in $pyProcs) {
        Watchdog-Log "  Training proc PID=$($p.Id) WS=$([math]::Round($p.WorkingSet64/1MB,1))MB CPU=$($p.CPU)"
    }
}

# Decide what to do
if ($batchAlive) {
    # Batch runner alive, nothing to do
    exit 0
}

if ($trainingRunning) {
    Watchdog-Log "Batch runner dead but training subprocess still running. Waiting for it to finish."
    # Don't start a new batch runner - it would conflict with the running training
    exit 0
}

# Neither batch runner nor training is running - restart batch runner
Watchdog-Log "Neither batch runner nor training running. Restarting batch runner."

# Clean old log files (so we can tell new output from old)
foreach ($f in @($stdoutLog, $stderrLog)) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

# Start batch runner via Start-Process (detached)
$proc = Start-Process -FilePath $batPath `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 5

if ($proc -and -not $proc.HasExited) {
    "$($proc.Id)" | Out-File -FilePath $pidFile -Encoding ascii
    Watchdog-Log "Restarted batch runner PID=$($proc.Id)"
} else {
    Watchdog-Log "FAILED to restart batch runner"
    if (Test-Path $stderrLog) {
        $errContent = Get-Content $stderrLog -Tail 10
        Watchdog-Log "stderr: $errContent"
    }
}
