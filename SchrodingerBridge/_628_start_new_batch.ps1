# Start new batch for remaining 84 destructive ablation experiments
# Uses Start-Process to survive SSH session disconnect
$ErrorActionPreference = 'Continue'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'

# First check if current batch (PID 16248) is still running
$batchRunning = $false
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
foreach ($p in $pyProcs) {
    if ($p.WorkingSet64 -gt 1GB) {
        $batchRunning = $true
        Write-Host "Training process still running: PID=$($p.Id) WS=$([math]::Round($p.WorkingSet64/1MB,1))MB"
        break
    }
}

if ($batchRunning) {
    Write-Host "Waiting for current batch to finish..."
    $waited = 0
    while ($batchRunning -and $waited -lt 600) {
        Start-Sleep -Seconds 30
        $waited += 30
        $batchRunning = $false
        $pyProcs = Get-Process python -ErrorAction SilentlyContinue
        foreach ($p in $pyProcs) {
            if ($p.WorkingSet64 -gt 1GB) {
                $batchRunning = $true
                break
            }
        }
        if ($waited % 120 -eq 0) {
            Write-Host "  Still waiting... ${waited}s elapsed"
        }
    }
    if ($batchRunning) {
        Write-Host "Timeout waiting for current batch. Proceeding anyway (may conflict)."
    } else {
        Write-Host "Current batch finished after ${waited}s."
    }
}

Write-Host "`n=== Starting new batch for remaining experiments ==="
$py = 'C:\Progra~1\Python312\python.exe'
$batchScript = "$root\628_run_destructive_batch.py"
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log_v2.txt"

# Start batch in background with Start-Process to survive SSH disconnect
$proc = Start-Process -FilePath $py -ArgumentList $batchScript -WorkingDirectory $root -RedirectStandardOutput $batchLog -RedirectStandardError "$batchLog.err" -WindowStyle Hidden -PassThru

Write-Host "New batch started: PID=$($proc.Id)"
Write-Host "Log: $batchLog"
Write-Host "Batch will run ~8h for 84 experiments (341s each)"

# Wait a bit and check it started
Start-Sleep -Seconds 10
if ($proc.HasExited) {
    Write-Host "ERROR: Batch process exited immediately with code $($proc.ExitCode)"
    Write-Host "=== Error log ==="
    if (Test-Path "$batchLog.err") { Get-Content "$batchLog.err" -Tail 20 }
} else {
    Write-Host "Batch process is running. First 5 lines of log:"
    if (Test-Path $batchLog) { Get-Content $batchLog -Head 5 }
}
