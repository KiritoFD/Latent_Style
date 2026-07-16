# Check if test dir exists and inspect main table image set
Write-Host "=== dataset root ==="
$ds = 'I:\datasets\wikiart_distinct5_samam_512_classview'
Get-ChildItem $ds -ErrorAction SilentlyContinue | Select-Object Name, Mode | Format-Table -AutoSize

Write-Host "=== test dir (main table style refs) ==="
$test = "$ds\test"
if (Test-Path $test) {
    Write-Host "EXISTS: $test"
    Get-ChildItem $test | Select-Object Name | Format-Table -AutoSize
    $firstStyle = Get-ChildItem $test | Select-Object -First 1
    if ($firstStyle.PSIsContainer) {
        $cnt = (Get-ChildItem $firstStyle.FullName -File).Count
        Write-Host "  $($firstStyle.Name): $cnt files"
    }
} else {
    Write-Host "NOT FOUND: $test"
}

Write-Host "=== train dir (curve script used this) ==="
$train = "$ds\train"
if (Test-Path $train) {
    Get-ChildItem $train | Select-Object Name | Format-Table -AutoSize
    $firstStyle = Get-ChildItem $train | Select-Object -First 1
    if ($firstStyle.PSIsContainer) {
        $cnt = (Get-ChildItem $firstStyle.FullName -File).Count
        Write-Host "  $($firstStyle.Name): $cnt files"
    }
}

Write-Host "=== main table samam images ==="
$mainImgs = 'I:\Github\Latent_Style\SchrodingerBridge\results\D5-512\samam'
if (Test-Path $mainImgs) {
    $cnt = (Get-ChildItem $mainImgs -File).Count
    Write-Host "EXISTS: $mainImgs ($cnt files)"
    Get-ChildItem $mainImgs -File | Select-Object -First 5 | Select-Object Name | Format-Table -AutoSize
} else {
    Write-Host "NOT FOUND: $mainImgs"
    # check local path mapping
    $localMain = 'G:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\samam'
    Write-Host "(local path: $localMain not accessible from remote)"
}
