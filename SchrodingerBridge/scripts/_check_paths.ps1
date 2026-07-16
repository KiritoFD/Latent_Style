# Check dataset path layout on remote
Write-Host "=== I:\Github\Latent_Style\ contents ==="
Get-ChildItem 'I:\Github\Latent_Style' -ErrorAction SilentlyContinue | Select-Object Name, Mode, Target | Format-Table -AutoSize

Write-Host "=== I:\datasets\ top ==="
Get-ChildItem 'I:\datasets' -ErrorAction SilentlyContinue | Select-Object Name -First 10 | Format-Table -AutoSize

Write-Host "=== check if Dataset symlink exists under project root ==="
$ds = 'I:\Github\Latent_Style\Dataset'
if (Test-Path $ds) {
    $item = Get-Item $ds
    Write-Host "Exists: $ds"
    Write-Host "Mode: $($item.Mode)"
    Write-Host "Target: $($item.Target)"
    Write-Host "LinkType: $($item.LinkType)"
} else {
    Write-Host "Not found: $ds"
}

Write-Host "=== check wikiart_distinct5 under project root ==="
$ds2 = 'I:\Github\Latent_Style\Dataset\wikiart_distinct5_samam_512_classview'
if (Test-Path $ds2) {
    Write-Host "Exists: $ds2"
    Get-ChildItem $ds2 | Select-Object Name | Format-Table -AutoSize
} else {
    Write-Host "Not found: $ds2"
}

Write-Host "=== original dataset path ==="
$ds3 = 'I:\datasets\wikiart_distinct5_samam_512_classview'
if (Test-Path $ds3) {
    Write-Host "Exists: $ds3"
    Get-ChildItem $ds3 | Select-Object Name | Format-Table -AutoSize
} else {
    Write-Host "Not found: $ds3"
}
