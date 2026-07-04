# Check N1 stderr/stdout logs
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$errLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
$outLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stdout.log"

Write-Host "=== N1 stderr log ==="
if (Test-Path $errLog) {
    $size = (Get-Item $errLog).Length
    Write-Host "stderr size = $size bytes"
    if ($size -gt 0) {
        Write-Host "--- Full stderr content ---"
        Get-Content $errLog
    } else {
        Write-Host "[EMPTY]"
    }
} else {
    Write-Host "[NOT CREATED] stderr log not found"
}

Write-Host ""
Write-Host "=== N1 stdout log (last 50 lines) ==="
if (Test-Path $outLog) {
    $size = (Get-Item $outLog).Length
    Write-Host "stdout size = $size bytes"
    if ($size -gt 0) {
        Get-Content $outLog -Tail 50
    } else {
        Write-Host "[EMPTY]"
    }
} else {
    Write-Host "[NOT CREATED] stdout log not found"
}

Write-Host ""
Write-Host "=== Schtask Status ==="
$task = Get-ScheduledTask -TaskName "n1_train" -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "Task State: $($task.State)"
    $info = $task | Get-ScheduledTaskInfo
    Write-Host "Last Run Time: $($info.LastRunTime)"
    Write-Host "Last Task Result: 0x$('{0:X}' -f $info.LastTaskResult)"
}

Write-Host ""
Write-Host "=== Running Python Processes ==="
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "[OK] Found $($pythonProcs.Count) Python process(es):"
    $pythonProcs | ForEach-Object {
        $age = (Get-Date) - $_.StartTime
        Write-Host "  PID=$($_.Id) CPU=$($_.CPU)s Mem=$([math]::Round($_.WorkingSet64/1MB,1))MB Age=$([math]::Round($age.TotalSeconds, 0))s"
    }
} else {
    Write-Host "[WARN] No Python process running"
}
