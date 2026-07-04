# 628 hourly monitor: sleep 3600s then check progress and write report
param([int]$Round = 1)

Start-Sleep -Seconds 3600

$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$reportPath = "$root\exp\628_ablation\destructive_logs\hourly_report_r$Round.md"
$batchLog = "$root\exp\628_ablation\destructive_logs\batch_log.txt"
$watchdogLog = "$root\exp\628_ablation\destructive_logs\watchdog.log"
$cfgDir = "$root\configs\ablations\628_destructive"
$expDir = "$root\exp\628_ablation\destructive"
$pidFile = "$root\exp\628_ablation\destructive_logs\batch_runner.pid"

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

# Count progress
$allConfigs = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
$done = 0
$pending = 0
$doneList = @()
foreach ($cfg in $allConfigs) {
    $ep10 = Join-Path $expDir "$($cfg.BaseName)\epoch_0010.pt"
    if (Test-Path $ep10) {
        $done++
        $doneList += $cfg.BaseName
    } else {
        $pending++
    }
}

# Check processes
$pyProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 500MB }
$batchRunning = $false
if (Test-Path $pidFile) {
    $bpid = (Get-Content $pidFile -Raw).Trim()
    $p = Get-Process -Id $bpid -ErrorAction SilentlyContinue
    if ($p -and -not $p.HasExited) { $batchRunning = $true }
}

# GPU state
$gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Batch log tail
$batchTail = Get-Content $batchLog -Tail 5 -ErrorAction SilentlyContinue

# Build report
$report = @"
# 628 Hourly Monitor Report - Round $Round
Time: $ts

## Progress
- Done: $done / $($allConfigs.Count)
- Pending: $pending
- Batch runner running: $batchRunning
- Training processes: $($pyProcs.Count)

## GPU state
$gpu

## Batch log tail
$($batchTail -join "`n")

## Done experiments
$($doneList -join ", ")
"@

$report | Out-File -FilePath $reportPath -Encoding utf8
Write-Host $report
Write-Host "`nReport saved to: $reportPath"
