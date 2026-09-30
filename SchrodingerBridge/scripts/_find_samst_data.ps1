# Find and read SaMST curve CLIP-S/LPIPS data
$paths = @(
    'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206\eval_bundle\clip_lpips_curve.csv',
    'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206\eval_bundle',
    'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samst_wikiarts5_wsl_20260610_172206'
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Host "=== EXISTS: $p ==="
        if ((Get-Item $p).PSIsContainer) {
            Get-ChildItem $p -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize
        } else {
            Get-Content $p
        }
    } else {
        Write-Host "NOT FOUND: $p"
    }
}

# Also check for stepalign40 summary
Write-Host "`n=== Search for stepalign40 ==="
Get-ChildItem 'I:\Github\Latent_Style' -Recurse -Directory -Filter "*stepalign*" -ErrorAction SilentlyContinue -Depth 5 | Select-Object FullName | Format-Table -AutoSize

# Check baseline_reeval
Write-Host "`n=== baseline_reeval ==="
$reeval = 'I:\Github\Latent_Style\baseline_reeval'
if (Test-Path $reeval) {
    Get-ChildItem $reeval -Depth 2 | Select-Object FullName | Format-Table -AutoSize
} else {
    # Try other locations
    $locations = @(
        'I:\Github\Latent_Style\WEAVE\baseline_reeval',
        'I:\Github\Latent_Style\SchrodingerBridge\baseline_reeval',
        'I:\Github\Latent_Style\Related_Works\baseline_reeval'
    )
    foreach ($loc in $locations) {
        if (Test-Path $loc) {
            Write-Host "FOUND: $loc"
            Get-ChildItem $loc -Depth 2 | Select-Object FullName | Format-Table -AutoSize
        }
    }
}
