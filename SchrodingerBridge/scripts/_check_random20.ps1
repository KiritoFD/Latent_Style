$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== CHECK random20 dataset on remote I: drive ==="
$paths = @(
    "I:\Github\Latent_Style\Dataset\wikiart_random20_512",
    "I:\datasets\wikiart_random20_512",
    "I:\wikiart_random20_512",
    "I:\Github\Latent_Style\Dataset\wikiart_random20_512_latents_ema"
)
foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Output "EXISTS: $p"
        Get-ChildItem $p -Directory | Select-Object Name | Format-Table -AutoSize
    } else {
        Write-Output "MISSING: $p"
    }
}
Write-Output "=== CHECK random20 checkpoint on remote ==="
$ckptPaths = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\630_random20_heun_5ep\epoch_0005.pt"
)
foreach ($p in $ckptPaths) {
    if (Test-Path $p) {
        Write-Output "EXISTS: $p"
    } else {
        Write-Output "MISSING: $p"
    }
}
Write-Output "=== DONE ==="
