# Compare step 7000 vs step 20000 image filenames
$step7000 = 'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_007000\images'
$step20000 = 'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_020000\images'

$names7000 = (Get-ChildItem $step7000 -File).Name | Sort-Object
$names20000 = (Get-ChildItem $step20000 -File).Name | Sort-Object

Write-Host "step 7000: $($names7000.Count) files"
Write-Host "step 20000: $($names20000.Count) files"

# Check overlap
$overlap = $names7000 | Where-Object { $names20000 -contains $_ }
Write-Host "overlap: $($overlap.Count) / $($names7000.Count)"

# Show first 5 from each
Write-Host "`n=== step 7000 first 5 ==="
$names7000 | Select-Object -First 5
Write-Host "`n=== step 20000 first 5 ==="
$names20000 | Select-Object -First 5

# Check which source images are used
Write-Host "`n=== step 7000 unique src stems ==="
$src7000 = $names7000 | ForEach-Object { ($_ -split '__to__')[0] } | Sort-Object -Unique
Write-Host "  $($src7000.Count) unique srcs"
Write-Host "  first 3: $($src7000 | Select-Object -First 3)"

Write-Host "`n=== step 20000 unique src stems ==="
$src20000 = $names20000 | ForEach-Object { ($_ -split '__to__')[0] } | Sort-Object -Unique
Write-Host "  $($src20000.Count) unique srcs"
Write-Host "  first 3: $($src20000 | Select-Object -First 3)"

# Check if src images are from same set
$srcOverlap = $src7000 | Where-Object { $src20000 -contains $_ }
Write-Host "`nsrc overlap: $($srcOverlap.Count) / $($src7000.Count)"
