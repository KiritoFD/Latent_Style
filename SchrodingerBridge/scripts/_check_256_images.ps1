# Check existing 256 baseline images
$dirs = @(
    @{ name = "adain_256"; path = "I:\exp_256_photo2art\adain_256" },
    @{ name = "wct_256"; path = "I:\exp_256_photo2art\wct_256" },
    @{ name = "samst_256"; path = "I:\exp_256_photo2art\samst_256" },
    @{ name = "samam_256"; path = "I:\exp_256_photo2art\samam_256" },
    @{ name = "sdturbo_256"; path = "I:\exp_256_photo2art\sdturbo_256" },
    @{ name = "styleid_256"; path = "I:\exp_256_photo2art\styleid_256" },
    @{ name = "identity_256"; path = "I:\exp_256_photo2art\identity_256" },
    @{ name = "seedream_256"; path = "I:\exp_256_photo2art\seedream_256" }
)
foreach ($d in $dirs) {
    if (Test-Path $d.path) {
        $subDirs = Get-ChildItem $d.path -Directory -ErrorAction SilentlyContinue
        if ($subDirs.Count -gt 0) {
            foreach ($sd in $subDirs) {
                $cnt = (Get-ChildItem $sd.FullName -Filter *.png -ErrorAction SilentlyContinue).Count + (Get-ChildItem $sd.FullName -Filter *.jpg -ErrorAction SilentlyContinue).Count
                Write-Host "  $($d.name)/$($sd.Name): $cnt"
            }
        } else {
            $cnt = (Get-ChildItem $d.path -Filter *.png -ErrorAction SilentlyContinue).Count + (Get-ChildItem $d.path -Filter *.jpg -ErrorAction SilentlyContinue).Count
            Write-Host "  $($d.name): $cnt"
        }
    } else {
        Write-Host "  $($d.name): NOT EXIST"
    }
}

Write-Host "`n=== baseline_v2/images 256 subdirs ==="
$baseV2 = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
if (Test-Path $baseV2) {
    Get-ChildItem $baseV2 -Directory | ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count + (Get-ChildItem $_.FullName -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host "  $($_.Name): $cnt"
    }
}
