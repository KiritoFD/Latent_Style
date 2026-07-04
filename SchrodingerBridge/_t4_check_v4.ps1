Write-Host "=== QUEUE V3 LOG ==="
$qp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_queue_v3.log"
if (Test-Path $qp) { Get-Content $qp -Raw } else { Write-Host "NOT FOUND" }

Write-Host "`n=== ALL PROCESSES WITH 'python' OR 'powershell' ==="
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match "python|powershell" } | ForEach-Object {
    Write-Host "PID=$($_.Id) Name=$($_.ProcessName) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB StartTime=$($_.StartTime)"
}

Write-Host "`n=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
Write-Host "`n=== GPU PROCS ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits

Write-Host "`n=== T4_D3 LOG ==="
$d3log = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.log"
if (Test-Path $d3log) { Write-Host "Size: $((Get-Item $d3log).Length) B"; if ((Get-Item $d3log).Length -gt 0) { Get-Content $d3log -Tail 5 } } else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4_D3 ERR ==="
$d3err = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.err"
if (Test-Path $d3err) { Write-Host "Size: $((Get-Item $d3err).Length) B"; Get-Content $d3err -Tail 15 } else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4_D3 RESULT JSON ==="
$d3json = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.json"
if (Test-Path $d3json) { Write-Host "EXISTS"; Get-Content $d3json -Raw } else { Write-Host "NOT FOUND YET" }

Write-Host "=== DONE ==="
