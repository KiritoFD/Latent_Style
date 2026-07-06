# Check CUT 512 image naming format
$cutImg = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut"
Write-Host "=== First 10 CUT 512 images ==="
Get-ChildItem $cutImg | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.Name)" }

Write-Host "`n=== Total count ==="
$png = (Get-ChildItem $cutImg -Filter *.png).Count
$jpg = (Get-ChildItem $cutImg -Filter *.jpg).Count
Write-Host "  PNG: $png  JPG: $jpg  Total: $($png + $jpg)"
