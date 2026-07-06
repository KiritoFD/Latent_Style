# Check all baseline image counts
$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== baseline_v2/images (Distinct5-512) ==="
$b2 = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
foreach ($d in (Get-ChildItem $b2 -Directory)) {
    $cnt = (Get-ChildItem $d.FullName -Filter *.png).Count
    Write-Host "  $($d.Name): $cnt"
}

Write-Host "`n=== baseline_wikiarts20 (WikiArt-20-512) ==="
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
foreach ($d in (Get-ChildItem $bw -Directory)) {
    $imgDir = Join-Path $d.FullName "images"
    $cnt = 0
    if (Test-Path $imgDir) {
        $cnt = (Get-ChildItem $imgDir -Filter *.png).Count
    } else {
        $cnt = (Get-ChildItem $d.FullName -Filter *.png).Count
    }
    Write-Host "  $($d.Name): $cnt"
}

Write-Host "`n=== exp_256_photo2art (Photo2Art-256) ==="
$e256 = "I:\exp_256_photo2art"
if (Test-Path $e256) {
    foreach ($d in (Get-ChildItem $e256 -Directory)) {
        $imgDir = Join-Path $d.FullName "images"
        $cnt = 0
        if (Test-Path $imgDir) {
            $cnt = (Get-ChildItem $imgDir -Filter *.png).Count
        } else {
            $cnt = (Get-ChildItem $d.FullName -Filter *.png).Count
        }
        Write-Host "  $($d.Name): $cnt"
    }
}

Write-Host "`n=== Checking for CUT checkpoints ==="
$cutDirs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints",
    "I:\Github\Latent_Style\SchrodingerBridge\checkpoints",
    "I:\Github\Latent_Style\SchrodingerBridge\exp"
)
foreach ($cp in $cutDirs) {
    if (Test-Path $cp) {
        Write-Host "`n  $cp :"
        Get-ChildItem $cp -Recurse -Filter "*cut*" -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object { Write-Host "    $($_.FullName)" }
    }
}

Write-Host "`n=== Checking for SaMam checkpoints ==="
foreach ($cp in $cutDirs) {
    if (Test-Path $cp) {
        Get-ChildItem $cp -Recurse -Filter "*samam*" -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object { Write-Host "    $($_.FullName)" }
    }
}
