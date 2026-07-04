# Check N1 all epoch evaluation results
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$n1Dir = "$base\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25"

Write-Host "=== N1 All Epoch Evaluation Results ==="
Write-Host ""
Write-Host "Epoch | all_pairs_clip | transfer_clip | all_pairs_lpips | transfer_lpips | dist_to_0.74"
Write-Host "------|----------------|---------------|-----------------|----------------|-------------"

$bestClip = 0
$bestEpoch = ""

for ($ep = 1; $ep -le 8; $ep++) {
    $epochDir = "epoch_$( '{0:D4}' -f $ep )"
    $summaryFile = "$n1Dir\full_eval\$epochDir\summary.json"
    if (Test-Path $summaryFile) {
        $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
        $transfer = $summary.analysis.style_transfer_ability
        $allpairs = $summary.analysis.all_pairs_overview

        $apClip = [math]::Round($allpairs.clip_style, 4)
        $trClip = [math]::Round($transfer.clip_style, 4)
        $apLpips = [math]::Round($allpairs.content_lpips, 4)
        $trLpips = [math]::Round($transfer.content_lpips, 4)
        $dist = [math]::Round(0.74 - $apClip, 4)

        Write-Host "  $ep   |    $apClip     |    $trClip    |    $apLpips     |    $trLpips    |   $dist"

        if ($apClip -gt $bestClip) {
            $bestClip = $apClip
            $bestEpoch = $epochDir
        }
    } else {
        Write-Host "  $ep   | [eval not yet complete]"
    }
}

Write-Host ""
Write-Host "=== Best Epoch ==="
Write-Host "Best: $bestEpoch with all_pairs_clip = $bestClip"

Write-Host ""
Write-Host "=== Comparison with previous experiments ==="
Write-Host "T4_D1_dwt (infer):    0.7325 (best overall)"
Write-Host "T5_D4_u01_v3 (infer): 0.7323"
Write-Host "N11+N16 ep7 (train):  0.7315"
Write-Host "N5_lvl2_hh3 (infer):  0.7311"
Write-Host "T5 baseline (train):  0.7307"
Write-Host "N1 ep1 (train):       0.7207 (below N11+N16)"

Write-Host ""
Write-Host "=== Current Training Progress (stderr tail) ==="
$errLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
if (Test-Path $errLog) {
    $lines = Get-Content $errLog -Tail 5
    $lines | ForEach-Object {
        if ($_.Length -gt 200) {
            Write-Host $_.Substring(0, 200) + "..."
        } else {
            Write-Host $_
        }
    }
}
