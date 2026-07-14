$files = @(
    'src\spectral620.py',
    'src\spectral_bridge620.py',
    'src\spectral_losses620.py',
    'src\blocks620.py',
    'src\style_families.py',
    'src\config_schema.py',
    'src\style_encoder620.py',
    'src\model.py'
)
$total = 0
foreach ($f in $files) {
    if (Test-Path $f) {
        $c = (Get-Content $f | Measure-Object -Line).Lines
        $name = Split-Path $f -Leaf
        Write-Host "$name : $c"
        $total += $c
    } else {
        Write-Host "$f : MISSING"
    }
}
Write-Host "TOTAL : $total"
