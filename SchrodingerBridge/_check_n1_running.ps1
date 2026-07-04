# Check N1 training is actually running (no stdout log this time, check process + CSV log)
$base = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== Running Python Processes ==="
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs) {
    Write-Host "[OK] Found $($pythonProcs.Count) Python process(es):"
    $pythonProcs | ForEach-Object {
        $cpu = $_.CPU
        $mem = [math]::Round($_.WorkingSet64 / 1MB, 1)
        $age = (Get-Date) - $_.StartTime
        Write-Host "  PID=$($_.Id) CPU=${cpu}s Mem=${mem}MB Age=$([math]::Round($age.TotalSeconds, 0))s"
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
    Write-Host "Last Task Result: 0x$('{0:X}' -f $info.LastTaskResult)"
} else {
    Write-Host "[WARN] Task n1_train not found"
}

Write-Host ""
Write-Host "=== N1 Experiment Dir Check ==="
$n1Dir = "$base\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25"
if (Test-Path $n1Dir) {
    Write-Host "[OK] N1 experiment dir created: $n1Dir"
    Get-ChildItem $n1Dir -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Replace($n1Dir, ".")
        Write-Host "  $rel"
    }
} else {
    Write-Host "[INFO] N1 experiment dir not yet created (training may still be in init phase)"
}

Write-Host ""
Write-Host "=== GPU Status ==="
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-Host $gpuInfo
    } else {
        Write-Host "[WARN] nvidia-smi returned no output"
    }
} else {
    Write-Host "[WARN] nvidia-smi not available"
}
