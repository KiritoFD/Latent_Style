$t5Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\t5_b2v2_d2_d4"
Write-Host "=== T5 checkpoint directory ==="
if (Test-Path $t5Dir) {
    Get-ChildItem $t5Dir -Filter "*.pt" | Sort-Object Name | ForEach-Object {
        Write-Host $_.Name
    }
    Write-Host ""
    Write-Host "=== T5 full_eval directory ==="
    $evalDir = "$t5Dir\full_eval"
    if (Test-Path $evalDir) {
        Get-ChildItem $evalDir -Directory | Sort-Object Name | ForEach-Object {
            $summaryFile = "$evalDir\$($_.Name)\summary.json"
            if (Test-Path $summaryFile) {
                $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
                $transfer = $summary.analysis.style_transfer_ability
                $allpairs = $summary.analysis.all_pairs_overview
                $clip = [math]::Round($allpairs.clip_style, 4)
                $lpips = [math]::Round($allpairs.content_lpips, 4)
                Write-Host "$($_.Name): clip=$clip lpips=$lpips"
            } else {
                Write-Host "$($_.Name): [no summary.json]"
            }
        }
    } else {
        Write-Host "No full_eval directory"
    }
} else {
    Write-Host "T5 dir not found"
}
