$root = "I:\datasets\wikiarts15_512_test"
$total = 0
Get-ChildItem $root -Directory | ForEach-Object {
    $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
    Write-Output ("{0,-32} {1}" -f $_.Name, $cnt)
    $total += $cnt
}
Write-Output ("-" * 40)
Write-Output ("TOTAL_DIRS={0} TOTAL_FILES={1}" -f (Get-ChildItem $root -Directory).Count, $total)
