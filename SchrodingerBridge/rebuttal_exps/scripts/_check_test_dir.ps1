$testDir = "I:\Github\Latent_Style\WEAVE\data\test"
$styles = Get-ChildItem $testDir -Directory -ErrorAction SilentlyContinue
Write-Host "Test styles: $($styles.Count)"
foreach ($s in $styles) {
    $allImgs = Get-ChildItem "$($s.FullName)" -Include *.png,*.jpg,*.jpeg -File -Recurse -ErrorAction SilentlyContinue
    Write-Host "  $($s.Name): $($allImgs.Count) images"
    if ($allImgs.Count -gt 0) {
        Write-Host "    Example: $($allImgs[0].Name)"
    }
}
# Also check if test_dir has direct files
$directFiles = Get-ChildItem "$testDir\*.jpg" -ErrorAction SilentlyContinue
Write-Host "Direct JPGs in test root: $($directFiles.Count)"
$directPngs = Get-ChildItem "$testDir\*.png" -ErrorAction SilentlyContinue
Write-Host "Direct PNGs in test root: $($directPngs.Count)"
