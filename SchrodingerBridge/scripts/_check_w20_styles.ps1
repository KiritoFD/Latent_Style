# Check W20 image directory file count and style distribution
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== W20 sdturbo images ==="
$dir = "$REPO\exp\baseline_wikiarts20\sdturbo\images"
if (Test-Path $dir) {
    $files = Get-ChildItem $dir -File
    Write-Host "Total: $($files.Count)"
    # Parse target styles from filenames
    $styles = @{}
    foreach ($f in $files) {
        $name = $f.BaseName
        if ($name -match "_to_(.+)$") {
            $tgt = $matches[1]
            if (-not $styles.ContainsKey($tgt)) { $styles[$tgt] = 0 }
            $styles[$tgt]++
        }
    }
    Write-Host "Target styles:"
    $styles.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host "  $($_.Name): $($_.Value)" }
    Write-Host "Sample filenames:"
    $files | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host ""
Write-Host "=== D5 baseline_v2 adain images ==="
$dir2 = "$REPO\exp\baseline_v2\images\adain"
if (Test-Path $dir2) {
    $files2 = Get-ChildItem $dir2 -File
    Write-Host "Total: $($files2.Count)"
    Write-Host "Sample filenames:"
    $files2 | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host ""
Write-Host "=== D5 baseline_v2 wct_v32k images ==="
$dir3 = "$REPO\exp\baseline_v2\images\wct_v32k"
if (Test-Path $dir3) {
    $files3 = Get-ChildItem $dir3 -File
    Write-Host "Total: $($files3.Count)"
    Write-Host "Sample filenames:"
    $files3 | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
}
