# Kill stuck powershell queue processes (PIDs from launcher v3)
# Then verify _run_t4_infer.ps1 exists
Write-Host "=== KILLING STUCK QUEUE ==="
# Kill any powershell processes that are running the queue scripts (except this one and SSH)
Get-Process powershell -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Id -ne $PID) {
        Write-Host "Killing powershell PID=$($_.Id)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2

Write-Host "`n=== VERIFY _run_t4_infer.ps1 EXISTS ==="
$rp = "I:/Github/Latent_Style/SchrodingerBridge/_run_t4_infer.ps1"
if (Test-Path $rp) {
    Write-Host "EXISTS"
    Get-Content $rp
} else {
    Write-Host "NOT FOUND - need to recreate"
}

Write-Host "`n=== VERIFY _p4_infer_ablation.py EXISTS ==="
$ap = "I:/Github/Latent_Style/SchrodingerBridge/_p4_infer_ablation.py"
if (Test-Path $ap) { Write-Host "EXISTS" } else { Write-Host "NOT FOUND" }

Write-Host "`n=== VERIFY T4 CHECKPOINT EXISTS ==="
$cp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/epoch_0001.pt"
if (Test-Path $cp) { Write-Host "EXISTS Size=$([math]::Round((Get-Item $cp).Length/1MB,1))MB" } else { Write-Host "NOT FOUND" }

Write-Host "`n=== VERIFY T4 CONFIG EXISTS ==="
$cf = "I:/Github/Latent_Style/SchrodingerBridge/configs/p4_t4_full_fusion.json"
if (Test-Path $cf) { Write-Host "EXISTS" } else { Write-Host "NOT FOUND" }

Write-Host "`n=== CLEANUP DONE ==="
