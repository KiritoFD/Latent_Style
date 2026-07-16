$genDir = "I:\Github\Latent_Style\WEAVE\exp\repro_weave_d5"
$imgs = Get-ChildItem "$genDir\images\*.png" -ErrorAction SilentlyContinue
Write-Host "Generated images: $($imgs.Count)"
$csv = "$genDir\metrics.csv"
if (Test-Path $csv) {
    Write-Host "metrics.csv: EXISTS"
    $content = Get-Content $csv -TotalCount 2
    Write-Host "Header: $($content[0])"
    Write-Host "Row1: $($content[1])"
} else {
    Write-Host "metrics.csv: NOT FOUND"
}
# Also check test dir structure
$testDir = "I:\Github\Latent_Style\WEAVE\data\test"
$styles = Get-ChildItem $testDir -Directory -ErrorAction SilentlyContinue
Write-Host "Test styles: $($styles.Count)"
foreach ($s in $styles) {
    $refImgs = Get-ChildItem "$($s.FullName)\*.png" -ErrorAction SilentlyContinue
    Write-Host "  $($s.Name): $($refImgs.Count) refs"
}
