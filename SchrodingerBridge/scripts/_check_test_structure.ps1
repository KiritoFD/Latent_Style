# Check test dir structure and compare with train
Write-Host "=== I:\datasets\wikiart_distinct5_512_images\test\ ==="
$testDir = 'I:\datasets\wikiart_distinct5_512_images\test'
if (Test-Path $testDir) {
    Get-ChildItem $testDir | ForEach-Object {
        if ($_.PSIsContainer) {
            $cnt = (Get-ChildItem $_.FullName -File).Count
            Write-Host "  $($_.Name): $cnt files"
        }
    }
} else {
    Write-Host "NOT FOUND"
}

Write-Host "=== compare train dirs ==="
$train1 = 'I:\datasets\wikiart_distinct5_samam_512_classview\train\Ukiyo_e'
$train2 = 'I:\datasets\wikiart_distinct5_512_images\train\Ukiyo_e'
if ((Test-Path $train1) -and (Test-Path $train2)) {
    $cnt1 = (Get-ChildItem $train1 -File).Count
    $cnt2 = (Get-ChildItem $train2 -File).Count
    Write-Host "  samam_512_classview train Ukiyo_e: $cnt1"
    Write-Host "  512_images train Ukiyo_e: $cnt2"
    # Check if same files
    $names1 = (Get-ChildItem $train1 -File).Name | Sort-Object
    $names2 = (Get-ChildItem $train2 -File).Name | Sort-Object
    if ($names1 -eq $names2) {
        Write-Host "  SAME files"
    } else {
        Write-Host "  DIFFERENT files"
        $diff = Compare-Object $names1 $names2 -ErrorAction SilentlyContinue
        if ($diff) { Write-Host "  diff count: $($diff.Count)" }
    }
}

Write-Host "=== test dir file samples ==="
$ukiyoTest = 'I:\datasets\wikiart_distinct5_512_images\test\Ukiyo_e'
if (Test-Path $ukiyoTest) {
    Get-ChildItem $ukiyoTest -File | Select-Object -First 5 | Select-Object Name | Format-Table -AutoSize
}

Write-Host "=== check if test dir matches manifest test_sources ==="
$manifest = Get-Content 'I:\datasets\wikiart_distinct5_samam_512_classview\manifest.json' -Raw | ConvertFrom-Json
$ukiyoClass = $manifest.classes | Where-Object { $_.name -eq 'Ukiyo_e' }
$testSources = $ukiyoClass.test_sources
$testNames = $testSources | ForEach-Object { [System.IO.Path]::GetFileName($_) } | Sort-Object
$actualTest = (Get-ChildItem $ukiyoTest -File).Name | Sort-Object
Write-Host "manifest test count: $($testNames.Count)"
Write-Host "actual test count: $($actualTest.Count)"
$match = $testNames | Where-Object { $actualTest -contains $_ }
Write-Host "matching files: $($match.Count) / $($testNames.Count)"
