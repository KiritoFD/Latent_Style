# T4 state diagnostic - simple output, no quoting issues
Write-Host "=== PYTHON PROCS ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $mb = [math]::Round($_.WorkingSet64 / 1MB, 1)
    Write-Host "PID=$($_.Id) WS=${mb}MB"
}
Write-Host "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
Write-Host "=== T4 JSON RESULTS ==="
$dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
if (Test-Path $dir) {
    Get-ChildItem $dir -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "$($_.Name) | $($_.Length)B | $($_.LastWriteTime)"
    }
} else {
    Write-Host "DIR NOT FOUND: $dir"
}
Write-Host "=== T4 LOGS ==="
if (Test-Path $dir) {
    Get-ChildItem $dir -Filter "*.log" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "$($_.Name) | $($_.Length)B | $($_.LastWriteTime)"
    }
}
Write-Host "=== T4 EVAL SUBDIRS ==="
$t4root = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion"
if (Test-Path $t4root) {
    Get-ChildItem $t4root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "DIR: $($_.Name)"
    }
}
Write-Host "=== DONE ==="
