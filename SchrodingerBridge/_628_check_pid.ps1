# Check if PID 26752 (batch runner) is still running after 2 minutes
$pidToCheck = 26752
$p = Get-Process -Id $pidToCheck -ErrorAction SilentlyContinue
if ($p -and -not $p.HasExited) {
    Write-Host "PID $pidToCheck STILL RUNNING at $(Get-Date)"
    $wsMB = [math]::Round($p.WorkingSet64/1MB, 1)
    Write-Host "  WS=${wsMB}MB CPU=$($p.CPU)"
} else {
    Write-Host "PID $pidToCheck EXITED at $(Get-Date)"
    $stdoutLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_runner_stdout.log'
    $stderrLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_runner_stderr.log'
    if (Test-Path $stderrLog) {
        Write-Host "--- stderr (last 30 lines) ---"
        Get-Content $stderrLog -Tail 30
    }
    if (Test-Path $stdoutLog) {
        Write-Host "--- stdout (last 10 lines) ---"
        Get-Content $stdoutLog -Tail 10
    }
}

Write-Host "`n=== nvidia-smi ==="
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

Write-Host "`n=== All python processes ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,StartTime,CPU,@{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}

Write-Host "`n=== batch_log.txt tail ==="
$batchLog = 'I:/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/destructive_logs/batch_log.txt'
if (Test-Path $batchLog) {
    Get-Content $batchLog -Tail 5
}
