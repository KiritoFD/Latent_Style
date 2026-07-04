# Check N1 epoch_0001 evaluation results
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$n1Dir = "$base\exp\p4_fusion_breakout\n1_lvl2_gate03_whh25"

Write-Host "=== N1 epoch_0001 Evaluation Results ==="
$summaryFile = "$n1Dir\full_eval\epoch_0001\summary.json"
if (Test-Path $summaryFile) {
    $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json

    # Extract key metrics
    $analysis = $summary.analysis
    $transfer = $analysis.style_transfer_ability
    $allpairs = $analysis.all_pairs_overview
    $wfi = $summary.wfi_benchmark
    $genWfi = $wfi.generated_wfi

    Write-Host "Epoch 1 Metrics:"
    Write-Host "  transfer_clip_style = $($transfer.clip_style)"
    Write-Host "  transfer_content_lpips = $($transfer.content_lpips)"
    Write-Host "  all_pairs_clip_style = $($allpairs.clip_style)"
    Write-Host "  all_pairs_content_lpips = $($allpairs.content_lpips)"
    Write-Host "  wfi_score = $($genWfi.wfi_score)"
    Write-Host ""
    Write-Host "  Distance to 0.74 target: $([math]::Round(0.74 - $allpairs.clip_style, 4))"
} else {
    Write-Host "[WARN] summary.json not found: $summaryFile"
}

Write-Host ""
Write-Host "=== N1 curve_summary.json ==="
$curveFile = "$n1Dir\full_eval\curve_summary.json"
if (Test-Path $curveFile) {
    $curve = Get-Content $curveFile -Raw
    Write-Host $curve
}

Write-Host ""
Write-Host "=== Checkpoints so far ==="
$ckpts = Get-ChildItem $n1Dir -Filter "epoch_*.pt" -ErrorAction SilentlyContinue
if ($ckpts) {
    $ckpts | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1MB, 1))MB)" }
} else {
    Write-Host "  No checkpoints yet"
}

Write-Host ""
Write-Host "=== Current Training Progress (stderr tail) ==="
$errLog = "$base\exp\p4_fusion_breakout\n1_lvl2_stderr.log"
if (Test-Path $errLog) {
    # Get last line with Epoch progress
    $lines = Get-Content $errLog -Tail 5
    $lines | ForEach-Object {
        if ($_.Length -gt 200) {
            Write-Host $_.Substring(0, 200) + "..."
        } else {
            Write-Host $_
        }
    }
}
