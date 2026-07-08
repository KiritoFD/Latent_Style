$ErrorActionPreference = 'Continue'

Write-Host "============================================================"
Write-Host "PART A: Count SeeDream D5 images"
Write-Host "============================================================"
$sd_dir = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\images"
if (Test-Path $sd_dir) {
    $pngs = Get-ChildItem $sd_dir -Filter *.png -ErrorAction SilentlyContinue
    $cnt = $pngs.Count
    Write-Host "  SeeDream D5 png count: $cnt"
    Write-Host "  sample files:"
    $pngs | Select-Object -First 5 | ForEach-Object { Write-Host ("    " + $_.Name) }
    Write-Host "  ..."
    $pngs | Select-Object -Last 3 | ForEach-Object { Write-Host ("    " + $_.Name) }
} else {
    Write-Host "  NOT FOUND"
}

Write-Host "`n=== Check parent dir structure ==="
$sd_parent = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750"
if (Test-Path $sd_parent) {
    Get-ChildItem $sd_parent -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) { Write-Host ("  [DIR] " + $_.Name) }
        else { Write-Host ("  [FILE] " + $_.Name + " (" + $_.Length + " bytes)") }
    }
}

Write-Host "`n=== Check for any seedream summary/metrics JSON ==="
Get-ChildItem $sd_parent -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("--- " + $_.Name + " ---")
    Get-Content $_.FullName -TotalCount 50
    Write-Host ""
}

# Also check parent-parent for any results
$sd_pp = "I:\Github\Latent_Style\exp_baselines\seedream45_api"
if (Test-Path $sd_pp) {
    Write-Host "`n=== seedream45_api dir contents ==="
    Get-ChildItem $sd_pp -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            $sub_pngs = 0
            $sub_img = Join-Path $_.FullName "images"
            if (Test-Path $sub_img) {
                $sub_pngs = (Get-ChildItem $sub_img -Filter *.png -ErrorAction SilentlyContinue).Count
            }
            Write-Host ("  [DIR] " + $_.Name + " (images png=" + $sub_pngs + ")")
        } else {
            Write-Host ("  [FILE] " + $_.Name + " (" + $_.Length + " bytes)")
        }
    }
}

Write-Host "`n============================================================"
Write-Host "PART B: Check existing CUT P256 search scripts results"
Write-Host "============================================================"
# Check if there's any CUT 256 image anywhere
$cut256_dirs = @(
    "I:\Github\Latent_Style\exp_baselines\cut",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\pixel256_photo2art"
)
foreach ($d in $cut256_dirs) {
    if (Test-Path $d) {
        Write-Host "=== $d ==="
        Get-ChildItem $d -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.PSIsContainer) {
                $cnt = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
                if ($cnt -gt 0) { Write-Host ("  [DIR] " + $_.Name + " png=" + $cnt) }
            } elseif ($_.Extension -in @('.png','.jpg','.pth','.pt')) {
                Write-Host ("  [FILE] " + $_.Name + " (" + $_.Length + " bytes)")
            }
        }
    }
}

# Search for any cut 256 images in I:\Github\Latent_Style\final_works
Write-Host "`n=== final_works\CUT ==="
$fw_cut = "I:\Github\Latent_Style\final_works\CUT"
if (Test-Path $fw_cut) {
    Get-ChildItem $fw_cut -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host ("  " + $_.Name + " (" + $_.Length + " bytes)")
    }
}

Write-Host "`n============================================================"
Write-Host "PART C: Read CUT eval logs for clues"
Write-Host "============================================================"
$cut_logs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut_eval.log",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut_eval_err.log"
)
foreach ($l in $cut_logs) {
    if (Test-Path $l) {
        Write-Host "--- $l ---"
        Get-Content $l
        Write-Host ""
    }
}
