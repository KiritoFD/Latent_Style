$root = "g:\GitHub\Latent_Style\SchrodingerBridge\results"

Write-Host "============================================================"
Write-Host "Final Comprehensive Image Counts (png + jpg)"
Write-Host "============================================================"

$grand_total = 0
foreach ($ds in @("D5-512", "P256", "R5-WikiArt")) {
    Write-Host ""
    Write-Host "[$ds]"
    $ds_dir = Join-Path $root $ds
    if (Test-Path $ds_dir) {
        Get-ChildItem $ds_dir -Directory | Sort-Object Name | ForEach-Object {
            $png = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
            $jpg = (Get-ChildItem $_.FullName -Filter *.jpg -ErrorAction SilentlyContinue).Count
            $jpg += (Get-ChildItem $_.FullName -Filter *.jpeg -ErrorAction SilentlyContinue).Count
            $total = $png + $jpg
            $grand_total += $total
            $ext = if ($png -gt 0 -and $jpg -gt 0) { "png+jpg" } elseif ($png -gt 0) { "png" } elseif ($jpg -gt 0) { "jpg" } else { "-" }
            Write-Host ("  " + $_.Name.PadRight(10) + ": " + "$total".PadLeft(6) + " (${ext}: png=$png, jpg=$jpg)")
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Grand total: $grand_total images"
Write-Host "============================================================"

# Disk usage
$size = (Get-ChildItem $root -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeGB = [math]::Round($size / 1GB, 2)
Write-Host "Total size: $sizeGB GB"
