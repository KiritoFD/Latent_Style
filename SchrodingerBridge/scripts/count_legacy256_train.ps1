$styles = @('cezanne','Hayao','monet','photo','vangogh')
$total = 0
Write-Output "=== train ==="
foreach ($s in $styles) {
    $p = "G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\train\$s"
    if (Test-Path $p) {
        $c = (Get-ChildItem -Path $p -Filter *.jpg -File -ErrorAction SilentlyContinue).Count
        Write-Output "$s $c"
        $total += $c
    } else {
        Write-Output "$s MISSING"
    }
}
Write-Output "train_total $total"

$total2 = 0
Write-Output "=== test ==="
foreach ($s in $styles) {
    $p = "G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\test\$s"
    if (Test-Path $p) {
        $c = (Get-ChildItem -Path $p -Filter *.jpg -File -ErrorAction SilentlyContinue).Count
        Write-Output "$s $c"
        $total2 += $c
    } else {
        Write-Output "$s MISSING"
    }
}
Write-Output "test_total $total2"
