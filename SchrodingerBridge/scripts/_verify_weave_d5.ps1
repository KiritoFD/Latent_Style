$ErrorActionPreference = 'Continue'

Write-Host "=== WEAVE D5 images dir ==="
$d5_img_dir = 'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\images'
if (Test-Path $d5_img_dir) {
    $imgs = Get-ChildItem -Path $d5_img_dir -ErrorAction SilentlyContinue
    $pngs = $imgs | Where-Object { $_.Extension -eq '.png' }
    Write-Host ("  total_items: " + $imgs.Count)
    Write-Host ("  png_count: " + $pngs.Count)
    Write-Host "  sample files:"
    $pngs | Select-Object -First 5 | ForEach-Object { Write-Host ("    " + $_.Name) }
} else {
    Write-Host "  NOT FOUND - trying parent"
    $parent = 'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010'
    if (Test-Path $parent) {
        Get-ChildItem -Path $parent | ForEach-Object {
            if ($_.PSIsContainer) { Write-Host ("    [DIR] " + $_.Name) }
            else { Write-Host ("    [FILE] " + $_.Name + " (" + $_.Length + ")") }
        }
    }
}

Write-Host ""
Write-Host "=== WEAVE D5 summary.json (head) ==="
$sum = 'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json'
if (Test-Path $sum) {
    $content = Get-Content -Path $sum -TotalCount 80
    $content | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "=== Also check clean_base_v2_final (in case it's the WEAVE final) ==="
$cbf = 'I:\Github\Latent_Style\SchrodingerBridge\exp\629_subtractive\clean_base_v2_final\full_eval\epoch_0010\images'
if (Test-Path $cbf) {
    $imgs2 = Get-ChildItem -Path $cbf -ErrorAction SilentlyContinue
    $pngs2 = $imgs2 | Where-Object { $_.Extension -eq '.png' }
    Write-Host ("  png_count: " + $pngs2.Count)
    $pngs2 | Select-Object -First 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
}

Write-Host ""
Write-Host "=== W20 summary.json key fields ==="
$w20_sum = 'I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\summary.json'
if (Test-Path $w20_sum) {
    $content2 = Get-Content -Path $w20_sum -TotalCount 80
    $content2 | ForEach-Object { Write-Host $_ }
}
