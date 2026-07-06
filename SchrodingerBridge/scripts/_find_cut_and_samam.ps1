# Find CUT images and check SaMam existing 131 images
$ErrorActionPreference = "Continue"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== Find CUT images ==="
$cutLocations = @(
    "$REPO\exp\baseline_v2\images\cut",
    "$REPO\exp\baseline_wikiarts20\cut",
    "$REPO\exp\baseline_wikiarts20\cut\images",
    "I:\Github\Latent_Style\final_works\CUT"
)
foreach ($loc in $cutLocations) {
    if (Test-Path $loc) {
        $cnt = (Get-ChildItem $loc -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "  ${loc}: $cnt files"
    }
}

Write-Host ""
Write-Host "=== Check SaMam existing 131 images (which styles?) ==="
$samamDir = "$REPO\exp\baseline_wikiarts20\samam\images"
if (Test-Path $samamDir) {
    $files = Get-ChildItem $samamDir -File
    $styles = @{}
    foreach ($f in $files) {
        $name = $f.BaseName
        if ($name -match "_to_(.+)$") {
            $tgt = $matches[1]
            if (-not $styles.ContainsKey($tgt)) { $styles[$tgt] = 0 }
            $styles[$tgt]++
        }
    }
    Write-Host "  SaMam 131 images target styles:"
    $styles.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host "    $($_.Name): $($_.Value)" }
}

Write-Host ""
Write-Host "=== Check if CUT W20 images exist somewhere ==="
Get-ChildItem "$REPO\exp" -Recurse -Directory -Filter "cut*" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "  $($_.FullName): $cnt files"
    }
