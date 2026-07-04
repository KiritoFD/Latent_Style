Write-Host "=== TASKLIST python ==="
tasklist /fi "imagename eq python.exe" 2>$null
Write-Host "`n=== TASKLIST pythonw ==="
tasklist /fi "imagename eq pythonw.exe" 2>$null

Write-Host "`n=== WMIC python processes ==="
wmic process where "name like '%python%'" get processid,commandline,workingsetsize 2>$null

Write-Host "`n=== QUEUE LOG ==="
$qp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/_t4_queue_v3.log"
if (Test-Path $qp) { Get-Content $qp -Raw }

Write-Host "`n=== T4_D3 ERR (full) ==="
$d3err = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.err"
if (Test-Path $d3err) { Write-Host "Size: $((Get-Item $d3err).Length) B"; Get-Content $d3err }

Write-Host "`n=== T4_D3 LOG (full) ==="
$d3log = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D3_u01.log"
if (Test-Path $d3log) { Write-Host "Size: $((Get-Item $d3log).Length) B"; if ((Get-Item $d3log).Length -gt 0) { Get-Content $d3log } }

Write-Host "`n=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits

Write-Host "=== DONE ==="
