$files = @('src\spectral_losses620.py', 'src\spectral_bridge620.py', 'src\spectral620.py', 'src\blocks620.py')
$total = 0
foreach ($f in $files) {
    $c = (Get-Content $f).Count
    Write-Output "$f : $c"
    $total += $c
}
Write-Output "TOTAL: $total"
