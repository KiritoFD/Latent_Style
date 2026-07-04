$styles = @('cezanne','Hayao','monet','photo','vangogh')
foreach ($s in $styles) {
    $p = "g:\GitHub\Latent_Style\Dataset\legacy256_overfit50\train\$s"
    $c = (Get-ChildItem -Path $p -Filter *.jpg -File).Count
    Write-Output "$s $c"
}
