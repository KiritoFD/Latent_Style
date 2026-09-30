$log = 'C:\Users\Administrator\_samst_curve_clip_lpips.log'
if (Test-Path $log) {
    Get-Content $log
} else {
    Write-Host "no log"
}
$csv = 'I:\Github\Latent_Style\exp_samam\_dino_curve_repro\samst_curve_clip_lpips.csv'
if (Test-Path $csv) {
    Write-Host "=== CSV ==="
    Get-Content $csv
}
