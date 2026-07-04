# Poll for T4 result JSONs
$dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
$exps = @("T4_D0_baseline", "T4_D1_dwt", "T4_D2_u005", "T4_D3_u01", "T4_D4_u01_v3", "T4_D5_u005_v3", "T4_D6_u02_v3")

Write-Host "=== T4 RESULT STATUS $(Get-Date -Format 'HH:mm:ss') ==="
foreach ($exp in $exps) {
    $jp = Join-Path $dir "$exp.json"
    if (Test-Path $jp) {
        $j = Get-Content $jp -Raw | ConvertFrom-Json
        $m = $j.metrics
        $p = $j.params
        if ($m.transfer_clip_style -ne $null) {
            Write-Host "DONE $exp | a=$($p.style_extrap_alpha) k=$($p.patch_adain_kernel) lp=$($p.lowpass_mode) | t_clip=$([math]::Round($m.transfer_clip_style,4)) t_lpips=$([math]::Round($m.transfer_content_lpips,4)) ap_clip=$([math]::Round($m.allpairs_clip_style,4)) ap_lpips=$([math]::Round($m.allpairs_content_lpips,4))"
        } else {
            Write-Host "PARTIAL $exp (JSON exists but no metrics)"
        }
    } else {
        # Check log size for progress
        $lp = Join-Path $dir "$exp.log"
        $sz = 0
        if (Test-Path $lp) { $sz = (Get-Item $lp).Length }
        Write-Host "PENDING $exp (log: $sz B)"
    }
}

Write-Host "`n=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits

Write-Host "`n=== PYTHON PROCS ==="
$pyprocs = Get-Process python -ErrorAction SilentlyContinue
if ($pyprocs) {
    $pyprocs | ForEach-Object { Write-Host "PID=$($_.Id) WS=$([math]::Round($_.WorkingSet64/1MB,1))MB" }
} else {
    Write-Host "No python processes"
}

Write-Host "=== DONE ==="
