# Check N1 training status
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$logFile = "$base\exp\p4_fusion_breakout\n1_lvl2_train.log"

Write-Host "=== N1 Training Log Check ==="
if (Test-Path $logFile) {
    $size = (Get-Item $logFile).Length
    Write-Host "[OK] Log file exists, size = $size bytes"
    if ($size -gt 0) {
        Write-Host "--- Last 40 lines of log ---"
        Get-Content $logFile -Tail 40
    } else {
        Write-Host "[WARN] Log file is empty"
    }
} else {
    Write-Host "[WARN] Log file does not exist yet: $logFile"
    Write-Host "Checking exp/p4_fusion_breakout/ for any n1 logs:"
    $parent = "$base\exp\p4_fusion_breakout"
    if (Test-Path $parent) {
        Get-ChildItem $parent -Filter "*n1*" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
        Get-ChildItem $parent -Filter "*.log" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name) (size=$($_.Length))" }
    }
}

Write-Host ""
Write-Host "=== Running Python Processes ==="
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "[OK] Found $($pythonProcs.Count) Python process(es):"
    $pythonProcs | ForEach-Object {
        $cpu = $_.CPU
        $mem = [math]::Round($_.WorkingSet64 / 1MB, 1)
        Write-Host "  PID=$($_.Id) CPU=${cpu}s Mem=${mem}MB StartTime=$($_.StartTime)"
    }
} else {
    Write-Host "[WARN] No Python process running"
}

Write-Host ""
Write-Host "=== Schtask Status ==="
$task = Get-ScheduledTask -TaskName "n1_train" -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "Task State: $($task.State)"
    $info = $task | Get-ScheduledTaskInfo
    Write-Host "Last Run Time: $($info.LastRunTime)"
    Write-Host "Last Task Result: $($info.LastTaskResult)"
    Write-Host "Next Run Time: $($info.NextRunTime)"
} else {
    Write-Host "[WARN] Task n1_train not found"
}
