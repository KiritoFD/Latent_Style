$remote = 'I:\wikiart_distinct5_samam_512_classview\test'
$styles = @('Early_Renaissance','Impressionism','Minimalism','Rococo','Ukiyo_e')
Write-Host "=== Remote test set ==="
$total = 0
foreach ($s in $styles) {
    $d = Join-Path $remote $s
    if (Test-Path $d) {
        $n = (Get-ChildItem $d -File | Where-Object { $_.Extension -match '\.(jpg|png|jpeg)' }).Count
        Write-Host "  $s : $n"
        $total += $n
    } else {
        Write-Host "  $s : MISS"
    }
}
Write-Host "  TOTAL: $total"
