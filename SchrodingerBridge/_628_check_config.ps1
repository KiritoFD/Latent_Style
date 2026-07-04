$cfgPath = 'I:\Github\Latent_Style\SchrodingerBridge\configs\ablations\628_destructive\X10_contrast_w10.json'
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json

Write-Host "=== Training config ==="
$cfg.training | Format-List

Write-Host ""
Write-Host "=== Checkpoint config ==="
$cfg.checkpoint | Format-List

Write-Host ""
Write-Host "=== Bridge config (loss weights) ==="
$cfg.bridge | Format-List

Write-Host ""
Write-Host "=== Eval-related keys ==="
$cfg.PSObject.Properties | Where-Object { $_.Name -match 'eval|full|probe' } | ForEach-Object {
    Write-Host "$($_.Name) = $($_.Value)"
}
