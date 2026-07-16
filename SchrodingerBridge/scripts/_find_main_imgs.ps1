# Search for main table SaMam images on remote
Write-Host "=== SchrodingerBridge\results ==="
$res = 'I:\Github\Latent_Style\SchrodingerBridge\results'
if (Test-Path $res) {
    Get-ChildItem $res -Recurse -Depth 2 | Select-Object FullName | Format-Table -AutoSize
} else {
    Write-Host "NOT FOUND: $res"
}

Write-Host "=== baseline_pipeline results ==="
$bp = 'I:\Github\Latent_Style\Related_Works\baseline_pipeline\results'
if (Test-Path $bp) {
    Get-ChildItem $bp -Depth 1 | Select-Object FullName | Format-Table -AutoSize
} else {
    Write-Host "NOT FOUND: $bp"
}

Write-Host "=== search for samam result dirs ==="
Get-ChildItem 'I:\Github\Latent_Style' -Recurse -Directory -Filter "*samam*" -ErrorAction SilentlyContinue -Depth 4 | Select-Object FullName | Format-Table -AutoSize

Write-Host "=== manifest.json ==="
$man = 'I:\datasets\wikiart_distinct5_samam_512_classview\manifest.json'
if (Test-Path $man) {
    Get-Content $man
} else {
    Write-Host "NOT FOUND: $man"
}
