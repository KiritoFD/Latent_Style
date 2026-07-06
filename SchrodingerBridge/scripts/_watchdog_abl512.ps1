# Watchdog script: checks if abl512 training is running, restarts if not
# Designed to be called by schtasks every 5 minutes
# Logic:
#   1. If python.exe is running abl512 config -> exit (training in progress)
#   2. If all 48 experiments have summary.json -> disable task (done)
#   3. Otherwise -> launch batch training script

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$BATCH_SCRIPT = "$REPO\scripts\run_abl512_v3.ps1"
$EXP_ROOT = "$REPO\exp\abl512"
$BATCH_LOG = "$REPO\logs\abl512_v3_batch.log"
$WATCHDOG_LOG = "$REPO\logs\abl512_watchdog.log"

function Write-WatchdogLog {
    param([string]$Msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $Msg" | Tee-Object -FilePath $WATCHDOG_LOG -Append
}

# 1. Check if python.exe is running with abl512 config
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    # Check if any python process has abl512 in its command line
    foreach ($proc in $pythonProcs) {
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($cmd -match "abl512") {
                Write-WatchdogLog "RUNNING: python PID $($proc.Id) is training abl512. Exit watchdog."
                exit 0
            }
        } catch {}
    }
}

# 2. Check if batch training script itself is running (powershell process)
$psProcs = Get-Process powershell -ErrorAction SilentlyContinue
foreach ($proc in $psProcs) {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
        if ($cmd -match "run_abl512_v3.ps1") {
            Write-WatchdogLog "RUNNING: powershell PID $($proc.Id) is running batch script. Exit watchdog."
            exit 0
        }
    } catch {}
}

# 3. Count completed experiments
$completed = 0
$total = 48
if (Test-Path $EXP_ROOT) {
    $summaries = Get-ChildItem -Path $EXP_ROOT -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue
    if ($summaries) {
        $completed = $summaries.Count
    }
}

if ($completed -ge $total) {
    Write-WatchdogLog "DONE: All $total experiments completed. Disabling watchdog task."
    # Disable the schtasks task
    schtasks /Change /TN "abl512_watchdog" /Disable
    exit 0
}

# 4. Launch batch training script
Write-WatchdogLog "RESTART: No abl512 training running. Completed=$completed/$total. Launching batch script."
$process = Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$BATCH_SCRIPT`"" -WindowStyle Hidden -PassThru
Write-WatchdogLog "  Launched PID: $($process.Id)"
Start-Sleep -Seconds 5
if (-not $process.HasExited) {
    Write-WatchdogLog "  Status: RUNNING"
} else {
    Write-WatchdogLog "  Status: FAILED to start (exit code $($process.ExitCode))"
}
