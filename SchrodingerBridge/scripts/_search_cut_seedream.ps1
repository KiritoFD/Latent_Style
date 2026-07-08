$ErrorActionPreference = 'Continue'

$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"

Write-Host "============================================================"
Write-Host "PART A: Search for Seedream images everywhere (depth 4)"
Write-Host "============================================================"
Get-ChildItem $exp_root -Directory -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream|see_dream|see-dream" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    $jpeg = (Get-ChildItem $d -Filter *.jpeg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs + " jpeg=" + $jpeg)
    if ($pngs -gt 0) {
        Get-ChildItem $d -Filter *.png | Select-Object -First 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

Write-Host "`n=== Also search I:\ root for seedream ==="
Get-ChildItem "I:\" -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
}

Write-Host "`n=== Search C:\ for seedream (limit) ==="
Get-ChildItem "C:\Users\Administrator" -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
}

Write-Host "`n============================================================"
Write-Host "PART B: Search for CUT images everywhere (depth 4)"
Write-Host "============================================================"
Get-ChildItem $exp_root -Directory -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^cut$|^cut_|_cut$|cut_photo|cut_photo2art" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
    if ($pngs -gt 0) {
        Get-ChildItem $d -Filter *.png | Select-Object -First 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

Write-Host "`n=== Also search I:\ root for cut ==="
Get-ChildItem "I:\" -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^cut$" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
}

Write-Host "`n============================================================"
Write-Host "PART C: Check baseline_v2 eval folder contents (all methods)"
Write-Host "============================================================"
$bv2_eval = "$exp_root\baseline_v2\eval"
if (Test-Path $bv2_eval) {
    Get-ChildItem $bv2_eval -Directory | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host ("  " + $_.Name + ": png=" + $pngs + " jpg=" + $jpgs)
    }
}

Write-Host "`n=== baseline_v2 images folder ==="
$bv2_img = "$exp_root\baseline_v2\images"
if (Test-Path $bv2_img) {
    Get-ChildItem $bv2_img -Directory | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host ("  " + $_.Name + ": png=" + $pngs + " jpg=" + $jpgs)
    }
}

Write-Host "`n============================================================"
Write-Host "PART D: Search for any 'photo2art' or '256' image dirs"
Write-Host "============================================================"
Get-ChildItem $exp_root -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "photo2art|p256|256" } | ForEach-Object {
    $d = $_.FullName
    $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    if ($pngs -gt 0 -or $jpgs -gt 0) {
        Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
    }
}

Write-Host "`n============================================================"
Write-Host "PART E: Check pixel256_photo2art and latent256_photo2art"
Write-Host "============================================================"
$p256 = "$exp_root\pixel256_photo2art"
if (Test-Path $p256) {
    Write-Host "=== pixel256_photo2art ==="
    Get-ChildItem $p256 -Directory -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
    }
}

$l256 = "$exp_root\latent256_photo2art"
if (Test-Path $l256) {
    Write-Host "`n=== latent256_photo2art ==="
    Get-ChildItem $l256 -Directory -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $jpgs = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host ("  " + $d + " png=" + $pngs + " jpg=" + $jpgs)
    }
}

Write-Host "`n============================================================"
Write-Host "PART F: Check existing CUT and Seedream eval JSONs"
Write-Host "============================================================"
$cut_jsons = @(
    "$exp_root\_eval_cut_w20.json",
    "$exp_root\_eval_cut_w20_musiq.json",
    "$exp_root\_eval_cut_d5_musiq.json"
)
foreach ($j in $cut_jsons) {
    if (Test-Path $j) {
        Write-Host "--- $j ---"
        Get-Content $j
        Write-Host ""
    }
}

# Search for any seedream eval json
Write-Host "=== Search for seedream JSON files ==="
Get-ChildItem $exp_root -Filter "*.json" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream" } | ForEach-Object {
    Write-Host ("--- " + $_.FullName + " ---")
    Get-Content $_.FullName -TotalCount 30
    Write-Host ""
}

# Check baseline_reeval for cut/seedream dirs
Write-Host "`n=== baseline_reeval subdirs ==="
$br = "$exp_root\baseline_reeval"
if (Test-Path $br) {
    Get-ChildItem $br -Directory | ForEach-Object {
        $d = $_.FullName
        $pngs = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        $sub_img = Join-Path $d "images"
        $sub_pngs = 0
        if (Test-Path $sub_img) { $sub_pngs = (Get-ChildItem $sub_img -Filter *.png -ErrorAction SilentlyContinue).Count }
        Write-Host ("  " + $_.Name + ": root_png=" + $pngs + " images_png=" + $sub_pngs)
    }
}
