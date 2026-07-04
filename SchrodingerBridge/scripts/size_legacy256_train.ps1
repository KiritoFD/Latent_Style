$size = (Get-ChildItem 'G:\GitHub\Latent_Style\Dataset\legacy256_overfit50\train' -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Output ("Size_GB: " + [math]::Round($size, 2))
