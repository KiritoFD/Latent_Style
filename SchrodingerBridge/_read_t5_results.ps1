# Read all T5 results for α trend analysis (while waiting for T4_D3)
$dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/infer_ablation"
$t5files = @(
    "T5_D0_baseline.json",
    "T5_D1_dwt.json",
    "T5_D2_u005.json",
    "T5_D3_u01.json",
    "T5_D4_u01_v3.json",
    "T5_D5_u005_v3.json",
    "T5_D6_u02_v3.json"
)

foreach ($f in $t5files) {
    $p = Join-Path $dir $f
    if (Test-Path $p) {
        $j = Get-Content $p -Raw | ConvertFrom-Json
        $m = $j.metrics
        $params = $j.params
        Write-Host "$($j.exp_name) | alpha=$($params.style_extrap_alpha) kernel=$($params.patch_adain_kernel) lp=$($params.lowpass_mode) | t_clip=$($m.transfer_clip_style) t_lpips=$($m.transfer_content_lpips) ap_clip=$($m.allpairs_clip_style) ap_lpips=$($m.allpairs_content_lpips)"
    } else {
        Write-Host "$f NOT FOUND"
    }
}
Write-Host "=== DONE ==="
