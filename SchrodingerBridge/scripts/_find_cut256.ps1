# Find CUT 256 images
$ErrorActionPreference = "Continue"

Write-Host "=== Look for CUT 256 images ==="
$locations = @(
    "I:\Github\Latent_Style\final_works\CUT",
    "I:\Github\Latent_Style\Related_Works\repos\CUT",
    "I:\exp_256_photo2art\cut_256",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut"
)

foreach ($loc in $locations) {
    Write-Host "--- $loc ---"
    if (Test-Path $loc) {
        Get-ChildItem $loc -Recurse -File -ErrorAction SilentlyContinue |
            Group-Object Extension |
            Select-Object Name, Count |
            Format-Table -Auto
        $sample = Get-ChildItem $loc -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 5
        $sample | ForEach-Object { Write-Host "  $($_.FullName)" }
    } else {
        Write-Host "  NOT FOUND"
    }
}

Write-Host ""
Write-Host "=== final_works\CUT\summary.json key info ==="
$sum = "I:\Github\Latent_Style\final_works\CUT\summary.json"
if (Test-Path $sum) {
    $j = Get-Content $sum -Raw | ConvertFrom-Json
    Write-Host "Keys: $($j.PSObject.Properties.Name -join ', ')"
    if ($j.matrix_breakdown) {
        Write-Host "matrix_breakdown entries: $($j.matrix_breakdown.Count)"
    }
}

Write-Host ""
Write-Host "=== Phase 2 SaMam progress ==="
$imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images"
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir -File).Count
    Write-Host "samam_w20 images: $cnt / 750"
}
