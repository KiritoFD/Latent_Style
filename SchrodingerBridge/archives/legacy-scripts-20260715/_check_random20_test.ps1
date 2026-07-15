$ErrorActionPreference = "SilentlyContinue"
Write-Output "=== CHECK random20 test dir on remote I: drive ==="
$paths = @(
    "I:\Github\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images\test",
    "I:\datasets\wikiart_random20_512",
    "I:\Github\Latent_Style\Dataset\wikiart_random20_512"
)
foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Output "EXISTS: $p"
        Get-ChildItem $p -Directory | Select-Object Name | Format-Table -AutoSize
    } else {
        Write-Output "MISSING: $p"
    }
}
Write-Output "=== DONE ==="
