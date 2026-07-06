# Status check for pipeline_fill_main + post_pipeline_fill
$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== Status Check $(Get-Date) ==="

# Task states
foreach ($tn in @("pipeline_fill_main", "post_pipeline_fill")) {
    $task = Get-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue
    if ($task) {
        $info = $task | Get-ScheduledTaskInfo
        Write-Host "[$tn] State: $($task.State)  LastRun: $($info.LastRunTime)  Result: $($info.LastTaskResult)"
    }
}

# Python processes
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
Write-Host "`nPython Processes: $($pyProcs.Count)"
foreach ($p in $pyProcs) {
    $dur = (Get-Date) - $p.StartTime
    Write-Host "  PID=$($p.Id) CPU=$($p.CPU.ToString('F0'))s Mem=$([math]::Round($p.WorkingSet64/1MB,0))MB Started=$($p.StartTime.ToString('HH:mm:ss')) Dur=$([math]::Round($dur.TotalMinutes,1))min"
}

# post_pipeline log tail (most recent activity)
$postLog = "I:\Github\Latent_Style\SchrodingerBridge\logs\post_pipeline.log"
if (Test-Path $postLog) {
    $logInfo = Get-Item $postLog
    Write-Host "`n--- post_pipeline.log LastWrite: $($logInfo.LastWriteTime) Size=$($logInfo.Length)B ---"
    Get-Content $postLog -Tail 25
}

# Image counts
Write-Host "`n=== Image Counts ==="
$dirs = @(
    @{ name = "sdturbo_256"; path = "I:\exp_256_photo2art\sdturbo_256\images" },
    @{ name = "styleid_256"; path = "I:\exp_256_photo2art\styleid_256\images" },
    @{ name = "samst_w20"; path = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samst\images" },
    @{ name = "sdturbo_w20"; path = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\sdturbo\images" },
    @{ name = "styleid_w20"; path = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\styleid\images" },
    @{ name = "samam_w20"; path = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images" }
)
foreach ($d in $dirs) {
    $png = (Get-ChildItem $d.path -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpg = (Get-ChildItem $d.path -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host "  $($d.name): $($png + $jpg)/750"
}

# Results JSON
$resFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\_pipeline_fill_results.json"
if (Test-Path $resFile) {
    Write-Host "`n=== Results JSON ==="
    Get-Content $resFile
}

# Eval JSONs
Write-Host "`n=== Eval JSONs ==="
$evals = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_*.json" -ErrorAction SilentlyContinue
foreach ($e in $evals) {
    Write-Host "--- $($e.Name) ---"
    Get-Content $e.FullName -Raw
}

# nvidia-smi
Write-Host "`n=== GPU Status ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
