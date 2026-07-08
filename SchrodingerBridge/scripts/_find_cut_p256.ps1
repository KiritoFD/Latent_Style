$ErrorActionPreference = 'Continue'

Write-Host "============================================================"
Write-Host "Search for CUT P256 images on G: drive"
Write-Host "============================================================"

$g_root = "G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5"
if (Test-Path $g_root) {
    Write-Host "=== $g_root tree (depth 3) ==="
    Get-ChildItem $g_root -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        if ($pngs -gt 0 -or $jpgs -gt 0) {
            Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
            if ($pngs -gt 0) {
                Get-ChildItem $d -Filter *.png | Select-Object -First 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
            }
        }
    }
} else {
    Write-Host "  $g_root NOT FOUND"
}

# Also check raw_results_val directly
$rv = "G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\raw_results_val"
if (Test-Path $rv) {
    Write-Host "`n=== $rv direct contents ==="
    Get-ChildItem $rv -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            $cnt = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
            $cnt2 = (Get-ChildItem $_.FullName -Filter *.jpg -ErrorAction SilentlyContinue).Count
            Write-Host ("  [DIR] " + $_.Name + " png=" + $cnt + " jpg=" + $cnt2)
        } else {
            Write-Host ("  [FILE] " + $_.Name + " (" + $_.Length + " bytes)")
        }
    }
}

# Search broader for any 256 cut images
Write-Host "`n=== Search G:\GitHub for cut_5x5 or cut_256 ==="
Get-ChildItem "G:\GitHub\Latent_Style" -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "cut" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    if ($pngs -gt 0) {
        Write-Host ("  " + $d + " png=" + $pngs)
        Get-ChildItem $d -Filter *.png | Select-Object -First 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

# Check if there are any .jpg images (CUT P256 might use jpg)
Write-Host "`n=== Search for jpg images in cut_5x5 ==="
if (Test-Path $g_root) {
    $jpgs = Get-ChildItem $g_root -Filter *.jpg -Recurse -ErrorAction SilentlyContinue
    $cnt = $jpgs.Count
    Write-Host "  total jpg count: $cnt"
    if ($cnt -gt 0) {
        $jpgs | Select-Object -First 5 | ForEach-Object { Write-Host ("    " + $_.FullName + " (" + $_.Length + " bytes)") }
    }
}
