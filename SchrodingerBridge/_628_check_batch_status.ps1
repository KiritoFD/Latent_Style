# Check the status of the batch runner (PID=26040)
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$stdoutLog = "$root\exp\628_ablation\destructive_logs\batch_runner_stdout.log"
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

# Read PID
if (Test-Path $pidFile) {
    $pid_value = Get-Content $pidFile -Raw
    $pid_value = $pid_value.Trim()
    Write-Host "PID file content: $pid_value"
    $proc = Get-Process -Id $pid_value -ErrorAction SilentlyContinue
    if ($proc -and -not $proc.HasExited) {
        Write-Host "Process $pid_value is RUNNING (CPU=$($proc.CPU) WS=$([math]::Round($proc.WorkingSet64/1MB,1))MB)"
    } else {
        Write-Host "Process $pid_value is NOT RUNNING"
    }
} else {
    Write-Host "No PID file"
}

Write-Host "`n=== batch_runner_stdout.log tail ==="
if (Test-Path $stdoutLog) {
    Get-Content $stdoutLog -Tail 15
} else {
    Write-Host "  (file not yet created)"
}

Write-Host "`n=== batch_log.txt tail ==="
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 10
}

Write-Host "`n=== nvidia-smi ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

Write-Host "`n=== python procs ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,WorkingSet
