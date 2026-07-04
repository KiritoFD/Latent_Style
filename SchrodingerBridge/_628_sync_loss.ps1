Copy-Item /tmp/spectral_losses620.py 'I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py' -Force
$info = Get-Item 'I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py'
Write-Host "Synced spectral_losses620.py: $($info.Length) bytes"
# Verify the new debug print is there
$content = Get-Content 'I:\Github\Latent_Style\SchrodingerBridge\src\spectral_losses620.py' -Raw
if ($content -match '628-ALL-DEBUG') {
    Write-Host "OK: 628-ALL-DEBUG print found in code"
} else {
    Write-Host "ERROR: 628-ALL-DEBUG print NOT found"
}
