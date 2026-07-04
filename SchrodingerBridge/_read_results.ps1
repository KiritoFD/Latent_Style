$summaryPath = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json"
$summary = Get-Content $summaryPath -Raw | ConvertFrom-Json

Write-Output "=== Stage 1 Clean Base V2 Evaluation Results ==="
Write-Output "Checkpoint: $($summary.checkpoint)"
Write-Output "Timestamp: $($summary.timestamp)"
Write-Output ""

# Extract pool-level metrics
if ($summary.pool) {
    Write-Output "=== Pool-level Metrics ==="
    $pool = $summary.pool
    Write-Output "clip_style:    $($pool.clip_style)"
    Write-Output "clip_content:  $($pool.clip_content)"
    Write-Output "clip_dir:      $($pool.clip_dir)"
    Write-Output "lpips:         $($pool.lpips)"
    Write-Output "lpips_alex:    $($pool.lpips_alex)"
    Write-Output "fid:           $($pool.fid)"
    Write-Output "delta_fid:     $($pool.delta_fid)"
}

Write-Output ""
Write-Output "=== Per-style clip_style ==="
if ($summary.per_style) {
    $summary.per_style | ForEach-Object {
        Write-Output "  $($_.target_style): clip_style=$($_.clip_style) lpips=$($_.lpips)"
    }
}

# Save key metrics to a simple file for easy comparison
$outFile = "I:\Github\Latent_Style\SchrodingerBridge\stage1_results.txt"
$pool = $summary.pool
$line = "Stage1 CleanBaseV2 | clip_style=$($pool.clip_style) | lpips=$($pool.lpips) | lpips_alex=$($pool.lpips_alex) | clip_content=$($pool.clip_content) | fid=$($pool.fid)"
Set-Content -Path $outFile -Value $line
Write-Output ""
Write-Output "Saved to: $outFile"
