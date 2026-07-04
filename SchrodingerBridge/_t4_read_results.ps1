# Read T4 results and logs
Write-Host "=== T4_D4_u01_v3.json ==="
$p = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D4_u01_v3.json"
if (Test-Path $p) { Get-Content $p -Raw } else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4_D0_baseline.log (last 30 lines) ==="
$lp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D0_baseline.log"
if (Test-Path $lp) { Get-Content $lp -Tail 30 } else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4_D4_u01_v3.log (last 20 lines) ==="
$lp2 = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation/T4_D4_u01_v3.log"
if (Test-Path $lp2) { Get-Content $lp2 -Tail 20 } else { Write-Host "NOT FOUND" }

Write-Host "`n=== T4 baseline summary.json (for reference) ==="
$bp = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t4_full_fusion/full_eval/epoch_0001/summary.json"
if (Test-Path $bp) {
    $s = Get-Content $bp -Raw | ConvertFrom-Json
    $tr = $s.analysis.style_transfer_ability
    $ap = $s.analysis.all_pairs_overview
    Write-Host "transfer_clip=$($tr.clip_style) transfer_lpips=$($tr.content_lpips)"
    Write-Host "allpairs_clip=$($ap.clip_style) allpairs_lpips=$($ap.content_lpips)"
} else { Write-Host "NOT FOUND" }
Write-Host "`n=== DONE ==="
