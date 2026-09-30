# Check if F drive exists and has wikiart source data
Write-Host "=== Drives ==="
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root, Used, Free | Format-Table -AutoSize

Write-Host "=== F:\wikiart\wikiart exists? ==="
if (Test-Path 'F:\wikiart\wikiart') {
    Write-Host "YES"
    Get-ChildItem 'F:\wikiart\wikiart' | Select-Object Name | Format-Table -AutoSize
} else {
    Write-Host "NO - F drive not available"
}

Write-Host "=== check a test source path ==="
$testPath = 'F:\wikiart\wikiart\Ukiyo_e\utagawa-toyokuni_the-heian-courtier.jpg'
if (Test-Path $testPath) {
    Write-Host "Test source EXISTS: $testPath"
} else {
    Write-Host "Test source NOT found: $testPath"
}

# Check if test images can be found in train dir by filename
Write-Host "=== check if test images are in train dir ==="
$trainDir = 'I:\datasets\wikiart_distinct5_samam_512_classview\train\Ukiyo_e'
$testFile = 'utagawa-toyokuni_the-heian-courtier.jpg'
$found = Get-ChildItem $trainDir -Filter $testFile -ErrorAction SilentlyContinue
if ($found) {
    Write-Host "Found in train: $testFile"
} else {
    Write-Host "NOT in train: $testFile"
}

# Check test_sources overlap with train
Write-Host "=== check test/train overlap ==="
$manifest = Get-Content 'I:\datasets\wikiart_distinct5_samam_512_classview\manifest.json' -Raw | ConvertFrom-Json
$ukiyoClass = $manifest.classes | Where-Object { $_.name -eq 'Ukiyo_e' }
$testSources = $ukiyoClass.test_sources
$trainSources = $ukiyoClass.train_sources
Write-Host "Ukiyo_e: train=$($ukiyoClass.train), test=$($ukiyoClass.test)"
Write-Host "test_sources[0]: $($testSources[0])"
$testNames = $testSources | ForEach-Object { [System.IO.Path]::GetFileName($_) }
$trainNames = $trainSources | ForEach-Object { [System.IO.Path]::GetFileName($_) }
$overlap = $testNames | Where-Object { $trainNames -contains $_ }
Write-Host "Overlap (test files also in train): $($overlap.Count) / $($testNames.Count)"
