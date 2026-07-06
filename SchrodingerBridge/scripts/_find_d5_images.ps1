# Find Distinct5-512 baseline images
$ErrorActionPreference = "Continue"

Write-Host "=== baseline_v2/images subdirs ==="
$base = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
if (Test-Path $base) {
    Get-ChildItem $base -Directory | ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host ("{0}: {1} files" -f $_.Name, $cnt)
    }
}

Write-Host ""
Write-Host "=== exp/distinct5 images ==="
$d5 = "I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5"
if (Test-Path $d5) {
    Get-ChildItem $d5 -Directory | ForEach-Object {
        $imgs = Join-Path $_.FullName "images"
        if (Test-Path $imgs) {
            $cnt = (Get-ChildItem $imgs -File -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Host ("{0}: {1} files" -f $_.Name, $cnt)
        }
    }
}

Write-Host ""
Write-Host "=== exp/ root dirs ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Directory |
    Where-Object { $_.Name -like "*baseline*" -or $_.Name -like "*distinct*" -or $_.Name -like "*512*" } |
    Select-Object Name

Write-Host ""
Write-Host "=== Search for styleid distinct5 images ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Recurse -Directory -Filter "styleid*" -ErrorAction SilentlyContinue |
    Select-Object -First 5 FullName

Write-Host ""
Write-Host "=== Look for adain_512 images ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Recurse -Directory -Filter "adain*" -ErrorAction SilentlyContinue |
    Select-Object -First 5 FullName

Write-Host ""
Write-Host "=== Look for samam 512 images (not 256) ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp" -Recurse -Directory -Filter "samam*" -ErrorAction SilentlyContinue |
    Select-Object -First 10 FullName
