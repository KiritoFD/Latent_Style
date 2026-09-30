# Search for wikiart source data on all drives
Write-Host "=== E:\ top ==="
Get-ChildItem 'E:\' -ErrorAction SilentlyContinue | Select-Object Name | Format-Table -AutoSize

Write-Host "=== Search for test image on all drives ==="
$testFile = 'hiroshige_hakone-kosuizu.jpg'
$drives = @('C:\','D:\','E:\','F:\','G:\','H:\','I:\')
foreach ($drv in $drives) {
    if (Test-Path $drv) {
        $found = Get-ChildItem $drv -Recurse -Filter $testFile -ErrorAction SilentlyContinue -Depth 5 | Select-Object -First 1
        if ($found) {
            Write-Host "FOUND on $drv : $($found.FullName)"
        }
    }
}

Write-Host "=== check I:\datasets for wikiart folders ==="
Get-ChildItem 'I:\datasets' -Directory -ErrorAction SilentlyContinue | Select-Object Name | Format-Table -AutoSize

Write-Host "=== check for wikiart on E: ==="
$eWiki = Get-ChildItem 'E:\' -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like '*wiki*' }
if ($eWiki) {
    $eWiki | Select-Object FullName | Format-Table -AutoSize
} else {
    Write-Host "No wiki* folders on E:\"
}
