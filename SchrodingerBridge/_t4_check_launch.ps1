Write-Host "=== LAUNCHER LOG ==="
$lp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_launcher.log"
if (Test-Path $lp) { Get-Content $lp -Raw } else { Write-Host "NOT FOUND" }

Write-Host "`n=== QUEUE LOG ==="
$qp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_queue_v2.log"
if (Test-Path $qp) { Get-Content $qp -Raw } else { Write-Host "NOT FOUND" }

Write-Host "`n=== POWERSHELL PROCS ==="
Get-Process powershell -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "PID=$($_.Id) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB StartTime=$($_.StartTime)"
}

Write-Host "`n=== ALL PYTHON PROCS ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "PID=$($_.Id) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB StartTime=$($_.StartTime)"
}
Write-Host "=== DONE ==="
