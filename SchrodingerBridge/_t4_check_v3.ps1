Write-Host "=== LAUNCHER V3 LOG ==="
$lp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_launcher_v3.log"
if (Test-Path $lp) { Get-Content $lp -Raw } else { Write-Host "NOT FOUND: $lp" }

Write-Host "`n=== QUEUE V3 LOG ==="
$qp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_queue_v3.log"
if (Test-Path $qp) { Get-Content $qp -Raw } else { Write-Host "NOT FOUND: $qp" }

Write-Host "`n=== ALL POWERSHELL PROCS ==="
Get-Process powershell -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "PID=$($_.Id) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB StartTime=$($_.StartTime)"
}

Write-Host "`n=== ALL PYTHON PROCS ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "PID=$($_.Id) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB StartTime=$($_.StartTime)"
}

Write-Host "`n=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits

Write-Host "`n=== T4_D3 LOG (if exists) ==="
$d3log = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.log"
if (Test-Path $d3log) {
    $sz = (Get-Item $d3log).Length
    Write-Host "Size: $sz B"
    if ($sz -gt 0) { Get-Content $d3log -Tail 10 }
} else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4_D3 ERR (if exists) ==="
$d3err = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.err"
if (Test-Path $d3err) {
    $sz = (Get-Item $d3err).Length
    Write-Host "Size: $sz B"
    if ($sz -gt 0) { Get-Content $d3err -Tail 10 }
} else { Write-Host "NOT FOUND" }

Write-Host "=== DONE ==="
