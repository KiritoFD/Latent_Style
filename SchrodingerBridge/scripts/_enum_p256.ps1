$ErrorActionPreference = 'Continue'

Write-Host "=== latent256_photo2art ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\latent256_photo2art"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== pixel256_photo2art ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\pixel256_photo2art"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== baseline_wikiarts15_256 subdirs (recursive 1 level) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B /S /AD I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256"
Write-Host $ssh_out
