# Check existing 628 destructive results and batch log status
$ErrorActionPreference = 'Continue'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$expDir = "$root\exp\628_ablation\destructive"
$logDir = "$root\exp\628_ablation\destructive_logs"

Write-Host "=== Existing experiment results ==="
if (Test-Path $expDir) {
    $dirs = Get-ChildItem $expDir -Directory -ErrorAction SilentlyContinue
    Write-Host "Exp dirs count: $($dirs.Count)"
    $doneCount = 0
    $partialCount = 0
    foreach ($d in $dirs) {
        $ep10 = Join-Path $d.FullName 'epoch_0010.pt'
        $ep8 = Join-Path $d.FullName 'epoch_0008.pt'
        if (Test-Path $ep10) {
            $doneCount++
        } elseif (Test-Path $ep8) {
            $partialCount++
        }
    }
    Write-Host "Completed (ep10 exists): $doneCount"
    Write-Host "Partial (ep8-9 only): $partialCount"
    Write-Host "`nFirst 10 exp dirs:"
    $dirs | Select-Object -First 10 | ForEach-Object {
        $ep10 = Join-Path $_.FullName 'epoch_0010.pt'
        $sum10 = Join-Path $_.FullName 'full_eval\epoch_0010\summary.json'
        $status = if (Test-Path $ep10) { 'DONE' } else { 'PARTIAL/FAIL' }
        $metrics = ''
        if (Test-Path $sum10) {
            $j = Get-Content $sum10 -Raw | ConvertFrom-Json
            $ap = $j.analysis.all_pairs_overview
            $metrics = " clip=$($ap.clip_style) lpips=$($ap.content_lpips)"
        }
        Write-Host "  $($_.Name): $status$metrics"
    }
} else {
    Write-Host "Exp dir MISSING"
}

Write-Host "`n=== Batch log status ==="
if (Test-Path $logDir) {
    $batchLog = Join-Path $logDir 'batch_log.txt'
    if (Test-Path $batchLog) {
        Write-Host "Batch log exists, last 15 lines:"
        Get-Content $batchLog -Tail 15
        Write-Host "`nBatch log size: $((Get-Item $batchLog).Length) bytes"
    } else {
        Write-Host "Batch log MISSING"
    }
    $logs = Get-ChildItem $logDir -Filter '*.log' -ErrorAction SilentlyContinue
    Write-Host "Individual logs count: $($logs.Count)"
} else {
    Write-Host "Log dir MISSING"
}

Write-Host "`n=== run.py check ==="
$runPy = "$root\src\run.py"
if (Test-Path $runPy) {
    Write-Host "run.py EXISTS"
    $first10 = Get-Content $runPy -Head 10
    Write-Host "First 10 lines:"
    $first10 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "run.py MISSING"
}

Write-Host "`n=== GPU status ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv 2>&1
