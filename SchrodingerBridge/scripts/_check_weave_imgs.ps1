$ErrorActionPreference = 'Continue'
$paths = @(
    'I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\images',
    'I:\Github\Latent_Style\SchrodingerBridge\exp\630_phase4j1_dwt_route'
)
foreach ($p in $paths) {
    Write-Host "=== $p ==="
    if (Test-Path $p) {
        $items = Get-ChildItem -Path $p -ErrorAction SilentlyContinue
        $dirs = $items | Where-Object { $_.PSIsContainer }
        $files = $items | Where-Object { -not $_.PSIsContainer }
        Write-Host ("  dirs: " + $dirs.Count)
        $dirs | Select-Object -First 10 | ForEach-Object { Write-Host ("    " + $_.Name) }
        Write-Host ("  files: " + $files.Count)
        $files | Select-Object -First 5 | ForEach-Object { Write-Host ("    " + $_.Name) }
        $pngs = $files | Where-Object { $_.Extension -eq '.png' }
        Write-Host ("  pngs: " + $pngs.Count)
    } else {
        Write-Host "  NOT FOUND"
    }
}

Write-Host ""
Write-Host "=== Search for any WEAVE-related images in exp ==="
$candidates = @(
    'I:\Github\Latent_Style\SchrodingerBridge\exp\abl512',
    'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256',
    'I:\Github\Latent_Style\SchrodingerBridge\exp\629_subtractive',
    'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2',
    'I:\Github\Latent_Style\SchrodingerBridge\exp\cspb'
)
foreach ($p in $candidates) {
    Write-Host "=== $p ==="
    if (Test-Path $p) {
        $items = Get-ChildItem -Path $p -ErrorAction SilentlyContinue
        $dirs = $items | Where-Object { $_.PSIsContainer }
        $files = $items | Where-Object { -not $_.PSIsContainer }
        Write-Host ("  dirs: " + $dirs.Count)
        $dirs | Select-Object -First 10 | ForEach-Object { Write-Host ("    " + $_.Name) }
        Write-Host ("  files: " + $files.Count)
        $files | Select-Object -First 5 | ForEach-Object { Write-Host ("    " + $_.Name) }
    } else {
        Write-Host "  NOT FOUND"
    }
}
